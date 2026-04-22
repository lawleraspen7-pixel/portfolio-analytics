from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import math
import pandas as pd
import numpy as np
import yfinance as yf

SECRET = os.environ.get("SECRET", "dev-secret")

app = FastAPI()


class Position(BaseModel):
    row: Optional[int] = None
    ticker: str
    execute: str = ""
    own: float = 0.0
    price: float = 0.0
    targetShares: float = 0.0
    costBasis: float = 0.0
    purchaseDate: str = ""
    sector: str = ""
    vol: float = 0.0


class Snapshot(BaseModel):
    generatedAt: str
    positions: List[Position] = Field(default_factory=list)


# =========================
# Locked live config
# =========================
TOP_N = 2
MOMENTUM_LOOKBACK = 60
VOL_LOOKBACK = 20
TREND_LOOKBACK = 20
REGIME_MA_LOOKBACK = 200
DOWNLOAD_PERIOD = "12mo"

MIN_TRADE_DOLLARS = 500.0
MIN_SHARE_DELTA = 0.10
MIN_WEIGHT_CHANGE = 0.04
MIN_TRADE_PCT = 0.04

MIN_WEIGHT = 0.15
MAX_WEIGHT = 0.65
MAX_WEIGHT_SHIFT_PER_DAY = 0.20
TURNOVER_CAP = 0.25

NORMAL_INVESTED_WEIGHT = 1.0
RISK_OFF_INVESTED_WEIGHT = 0.70

LEVERAGED_ETF_CAP = 0.15
REQUIRE_FULL_LOOKBACK = True
USE_BENCHMARK_RELATIVE_FILTER = True
MIN_QUALIFIERS_TO_STAY_FULLY_INVESTED = 2

REGIME_BENCHMARKS = ["SPY", "QQQ"]
LEVERAGED_TICKERS = {"SOXL", "TQQQ", "SPXL", "TECL", "UPRO", "FNGU"}
SEMI_ETFS = {"SMH", "SOXL"}
SEMI_NAMES = {"NVDA", "AMD", "AVGO"}

HIGH_CONFIDENCE_UNIVERSE = {
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLI", "XLY", "SMH", "SOXX",
    "NVDA", "AMD", "MSFT", "AAPL", "AMZN", "META", "AVGO", "SOXL"
}


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-12 else default


def round_shares(x: float) -> float:
    return round(x, 3)


def clean_ticker(x: str) -> str:
    return (x or "").strip().upper()


def clean_sector(x: str) -> str:
    x = (x or "").strip()
    return x if x else "Unknown"


def annualized_vol_from_returns(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    return float(returns.std(ddof=0) * math.sqrt(252))


def build_price_frame(tickers: List[str]) -> pd.DataFrame:
    tickers = list(dict.fromkeys([t for t in tickers if t]))
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=tickers,
        period=DOWNLOAD_PERIOD,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if data is None or getattr(data, "empty", True):
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        close_frames = []
        for ticker in tickers:
            if (ticker, "Close") in data.columns:
                close_frames.append(data[(ticker, "Close")].rename(ticker))
        if not close_frames:
            return pd.DataFrame()
        close = pd.concat(close_frames, axis=1)
    else:
        if "Close" not in data.columns or len(tickers) != 1:
            return pd.DataFrame()
        close = data[["Close"]].rename(columns={"Close": tickers[0]})

    return close.sort_index().ffill().dropna(how="all")


def enough_history(series: pd.Series, min_len: int) -> bool:
    return int(series.dropna().shape[0]) >= min_len


def compute_regime(close: pd.DataFrame) -> Dict[str, Any]:
    out = {"riskOn": True, "details": {}, "failed": []}

    for bench in REGIME_BENCHMARKS:
        if bench not in close.columns:
            out["details"][bench] = "missing"
            out["failed"].append(bench)
            continue

        s = close[bench].dropna()
        if len(s) < REGIME_MA_LOOKBACK:
            out["details"][bench] = "insufficient_history"
            out["failed"].append(bench)
            continue

        ma = s.rolling(REGIME_MA_LOOKBACK).mean().iloc[-1]
        px = s.iloc[-1]
        risk_on = bool(px > ma)
        out["details"][bench] = {
            "price": float(px),
            "ma200": float(ma),
            "riskOn": risk_on,
        }
        if not risk_on:
            out["failed"].append(bench)

    out["riskOn"] = len(out["failed"]) == 0
    return out


def compute_model(close: pd.DataFrame, candidate_tickers: List[str]) -> Dict[str, Any]:
    min_history = max(MOMENTUM_LOOKBACK, VOL_LOOKBACK, TREND_LOOKBACK, REGIME_MA_LOOKBACK) + 5

    valid_tickers = []
    for t in candidate_tickers:
        if t in close.columns and (not REQUIRE_FULL_LOOKBACK or enough_history(close[t], min_history)):
            valid_tickers.append(t)

    regime = compute_regime(close)

    if not valid_tickers:
        return {
            "latest_scores": {},
            "latest_momentum": {},
            "latest_trend": {},
            "latest_vol": {},
            "selected": [],
            "valid_tickers": [],
            "regime": regime,
        }

    px = close[valid_tickers].copy()
    rets = px.pct_change()
    momentum = px / px.shift(MOMENTUM_LOOKBACK) - 1.0
    vol = rets.rolling(VOL_LOOKBACK).std(ddof=0)
    trend = px / px.shift(TREND_LOOKBACK) - 1.0

    score = momentum / vol.replace(0.0, np.nan)
    score = score.where(trend > 0)

    if USE_BENCHMARK_RELATIVE_FILTER and "SPY" in close.columns:
        spy_momentum = (close["SPY"] / close["SPY"].shift(MOMENTUM_LOOKBACK) - 1.0).iloc[-1]
        keep = momentum.iloc[-1] > spy_momentum
        score.iloc[-1] = score.iloc[-1].where(keep)

    latest_score_row = score.iloc[-1].dropna().sort_values(ascending=False)

    # pull extra names so overlap rule can remove components and still leave enough candidates
    selected = latest_score_row.head(max(TOP_N + 3, 5)).index.tolist()

    if any(t in selected for t in SEMI_ETFS):
        selected = [t for t in selected if t not in SEMI_NAMES]

    selected = selected[:TOP_N]

    return {
        "latest_scores": {k: float(v) for k, v in latest_score_row.to_dict().items() if pd.notna(v)},
        "latest_momentum": {k: float(v) for k, v in momentum.iloc[-1].to_dict().items() if pd.notna(v)},
        "latest_trend": {k: float(v) for k, v in trend.iloc[-1].to_dict().items() if pd.notna(v)},
        "latest_vol": {k: float(v) for k, v in vol.iloc[-1].to_dict().items() if pd.notna(v)},
        "selected": selected,
        "valid_tickers": valid_tickers,
        "regime": regime,
    }


def apply_leveraged_cap(ticker: str, weight: float, risk_on: bool) -> float:
    if ticker in LEVERAGED_TICKERS:
        if not risk_on:
            return 0.0
        return min(weight, LEVERAGED_ETF_CAP)
    return weight


def cap_and_renormalize(weights: Dict[str, float], selected: List[str], target_sum: float) -> Dict[str, float]:
    out = {k: max(0.0, v) for k, v in weights.items()}

    for k in list(out.keys()):
        if k not in selected:
            out[k] = 0.0

    if not selected or target_sum <= 0:
        return {k: 0.0 for k in out}

    min_sum = MIN_WEIGHT * len(selected)
    if min_sum > target_sum:
        floor = target_sum / len(selected)
        for t in selected:
            out[t] = max(out.get(t, 0.0), floor)
    else:
        for t in selected:
            out[t] = max(out.get(t, 0.0), MIN_WEIGHT)

    total = sum(out.values())
    if total > 0:
        scale = target_sum / total
        out = {k: v * scale for k, v in out.items()}

    for _ in range(10):
        capped = {}
        excess = 0.0
        uncapped = []

        for t in selected:
            w = out.get(t, 0.0)
            cap = LEVERAGED_ETF_CAP if t in LEVERAGED_TICKERS else MAX_WEIGHT
            if w > cap:
                capped[t] = cap
                excess += w - cap
            else:
                capped[t] = w
                uncapped.append(t)

        if excess <= 1e-12 or not uncapped:
            out.update(capped)
            break

        uncapped_total = sum(capped[t] for t in uncapped)
        if uncapped_total <= 0:
            out.update(capped)
            break

        for t in uncapped:
            capped[t] += excess * (capped[t] / uncapped_total)
        out.update(capped)

    total = sum(out.values())
    if total > 0:
        scale = target_sum / total
        out = {k: v * scale for k, v in out.items()}

    return out


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
def analyze(snapshot: Snapshot, x_analytics_secret: str = Header(default="")):
    if x_analytics_secret != SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    positions: List[Dict[str, Any]] = []
    for raw in snapshot.positions:
        ticker = clean_ticker(raw.ticker)
        if not ticker or ticker == "CASH":
            continue
        positions.append({
            "row": raw.row,
            "ticker": ticker,
            "own": max(0.0, float(raw.own or 0.0)),
            "price": max(0.0, float(raw.price or 0.0)),
            "costBasis": max(0.0, float(raw.costBasis or 0.0)),
            "sector": clean_sector(raw.sector),
        })

    total_value = sum(p["own"] * p["price"] for p in positions)

    if total_value <= 0:
        return {
            "metrics": {
                "positionCount": len(positions),
                "totalMarketValue": 0.0,
                "avgVol": 0.0,
                "buyCount": 0,
                "sellCount": 0,
                "actionableTradeCount": 0,
                "topBuys": "",
                "weakestNames": "",
                "portfolioFlags": "no_positions_or_zero_value"
            },
            "diagnostics": [],
            "suggestedTargets": [],
            "sectorWeights": [],
            "decisionTrades": [],
            "emailSummary": {
                "generatedAt": snapshot.generatedAt,
                "topActions": [],
                "summaryLines": ["No positions with positive market value."]
            }
        }

    current_weight_map = {
        p["ticker"]: safe_div(p["own"] * p["price"], total_value, 0.0)
        for p in positions
    }

    sector_values: Dict[str, float] = {}
    for p in positions:
        value = p["own"] * p["price"]
        sector_values[p["sector"]] = sector_values.get(p["sector"], 0.0) + value
    current_sector_weights = {
        s: safe_div(v, total_value, 0.0)
        for s, v in sector_values.items()
    }

    live_tickers = [p["ticker"] for p in positions if p["ticker"] in HIGH_CONFIDENCE_UNIVERSE]
    excluded_live = [p["ticker"] for p in positions if p["ticker"] not in HIGH_CONFIDENCE_UNIVERSE]

    download_tickers = list(dict.fromkeys(live_tickers + REGIME_BENCHMARKS))
    close = build_price_frame(download_tickers)
    model = compute_model(close, live_tickers)

    latest_scores = model["latest_scores"]
    latest_momentum = model["latest_momentum"]
    latest_trend = model["latest_trend"]
    latest_vol = model["latest_vol"]
    selected = model["selected"]
    valid_tickers = set(model["valid_tickers"])
    regime = model["regime"]
    risk_on = bool(regime.get("riskOn", False))

    portfolio_flags = []
    if excluded_live:
        portfolio_flags.append("non_universe_holdings_present")
    if len(valid_tickers) < len(live_tickers):
        portfolio_flags.append("missing_market_data")
    if not risk_on:
        portfolio_flags.append("risk_off_regime")
    if len(selected) < TOP_N:
        portfolio_flags.append("partial_cash_signal")
    if not selected:
        portfolio_flags.append("no_positive_trend_candidates")

    if risk_on:
        target_invested_weight = NORMAL_INVESTED_WEIGHT
    else:
        target_invested_weight = RISK_OFF_INVESTED_WEIGHT

    if len(selected) < MIN_QUALIFIERS_TO_STAY_FULLY_INVESTED:
        target_invested_weight = min(target_invested_weight, len(selected) / max(TOP_N, 1)) if selected else 0.0

    ideal_weights = {p["ticker"]: 0.0 for p in positions}
    if selected:
        equal_w = safe_div(target_invested_weight, len(selected), 0.0)
        for t in selected:
            ideal_weights[t] = equal_w

    adjusted = {}
    for p in positions:
        t = p["ticker"]
        adjusted[t] = apply_leveraged_cap(t, ideal_weights.get(t, 0.0), risk_on)

    adjusted = cap_and_renormalize(adjusted, selected, target_invested_weight)

    target_weights = {}
    for p in positions:
        t = p["ticker"]
        current_w = current_weight_map.get(t, 0.0)
        ideal_w = adjusted.get(t, 0.0)

        if t in selected:
            diff = ideal_w - current_w
            if abs(diff) <= MAX_WEIGHT_SHIFT_PER_DAY:
                shifted = ideal_w
            else:
                shifted = current_w + (MAX_WEIGHT_SHIFT_PER_DAY if diff > 0 else -MAX_WEIGHT_SHIFT_PER_DAY)
        else:
            shifted = 0.0

        shifted = apply_leveraged_cap(t, shifted, risk_on)
        target_weights[t] = max(0.0, shifted)

    total_target = sum(target_weights.values())
    if total_target > target_invested_weight and total_target > 0:
        scale = target_invested_weight / total_target
        target_weights = {k: v * scale for k, v in target_weights.items()}

    diagnostics = []
    for p in positions:
        t = p["ticker"]
        value = p["own"] * p["price"]
        weight = current_weight_map[t]
        sector_weight = current_sector_weights.get(p["sector"], 0.0)
        unrealized_pct = safe_div((p["price"] - p["costBasis"]), p["costBasis"], 0.0) if p["costBasis"] > 0 else 0.0

        m60 = latest_momentum.get(t)
        tr20 = latest_trend.get(t)
        v20 = latest_vol.get(t)
        score = latest_scores.get(t, 0.0)

        comments = []
        if t in selected:
            comments.append("selected_top_momentum")
        if t not in HIGH_CONFIDENCE_UNIVERSE:
            comments.append("outside_live_universe")
        if t not in valid_tickers and t in HIGH_CONFIDENCE_UNIVERSE:
            comments.append("missing_price_history")
        if tr20 is not None and tr20 <= 0:
            comments.append("trend_filter_failed")
        if t in LEVERAGED_TICKERS:
            comments.append("leveraged_ticker")
        if not risk_on:
            comments.append("risk_off_regime")
        if t in SEMI_NAMES and any(x in selected for x in SEMI_ETFS):
            comments.append("excluded_by_overlap_rule")
        if weight > 0.20:
            comments.append("oversized")
        if sector_weight > 0.35:
            comments.append("crowded_sector")

        diagnostics.append({
            "row": p["row"],
            "ticker": t,
            "sector": p["sector"],
            "signalScore": round(float(score), 4),
            "convictionScore": round(float(score), 4),
            "riskScore": round(float(v20) if v20 is not None else 0.0, 4),
            "weight": round(weight, 4),
            "sectorWeight": round(sector_weight, 4),
            "currentDollarValue": round(value, 2),
            "unrealizedPct": round(unrealized_pct, 4),
            "gapShares": round((target_weights.get(t, 0.0) * total_value / p["price"] - p["own"]) if p["price"] > 0 else 0.0, 4),
            "gapValue": round(target_weights.get(t, 0.0) * total_value - value, 2),
            "comment": ", ".join(comments),
            "momentum60d": round(float(m60), 4) if m60 is not None else None,
            "trend20d": round(float(tr20), 4) if tr20 is not None else None,
            "vol20d": round(float(v20), 4) if v20 is not None else None,
        })

    diagnostics.sort(key=lambda d: d["convictionScore"], reverse=True)

    suggested_targets = []
    for p in positions:
        t = p["ticker"]
        suggested_targets.append({
            "ticker": t,
            "sector": p["sector"],
            "suggestedTargetWeight": round(target_weights.get(t, 0.0), 4),
            "convictionScore": round(float(latest_scores.get(t, 0.0)), 4)
        })
    suggested_targets.sort(key=lambda d: d["suggestedTargetWeight"], reverse=True)

    decision_trades = []
    raw_priorities = {}
    for p in positions:
        t = p["ticker"]
        price = p["price"]
        current_value = p["own"] * price
        current_weight = current_weight_map[t]
        suggested_weight = target_weights.get(t, 0.0)
        suggested_dollar_target = suggested_weight * total_value
        delta_dollar = suggested_dollar_target - current_value
        suggested_shares = safe_div(suggested_dollar_target, price, 0.0) if price > 0 else 0.0
        share_change = suggested_shares - p["own"]
        weight_change = suggested_weight - current_weight

        action_final = "HOLD"
        if abs(delta_dollar) >= total_value * MIN_TRADE_PCT:
            if abs(share_change) >= MIN_SHARE_DELTA and (abs(delta_dollar) >= MIN_TRADE_DOLLARS or abs(weight_change) >= MIN_WEIGHT_CHANGE):
                action_final = "BUY" if share_change > 0 else "SELL"

        conviction_score = float(latest_scores.get(t, 0.0))
        raw_priority = abs(delta_dollar) * (1.0 + abs(conviction_score))
        raw_priorities[t] = raw_priority

        decision_trades.append({
            "ticker": t,
            "sector": p["sector"],
            "currentShares": round(p["own"], 4),
            "currentDollarValue": round(current_value, 2),
            "suggestedTargetWeight": round(suggested_weight, 4),
            "suggestedDollarTarget": round(suggested_dollar_target, 2),
            "deltaDollar": round(delta_dollar, 2),
            "suggestedShares": round_shares(suggested_shares),
            "shareChange": round_shares(share_change),
            "actionFinal": action_final,
            "priorityScore": 0.0,
        })

    total_abs_trade = sum(abs(t["deltaDollar"]) for t in decision_trades)
    max_abs_trade = total_value * TURNOVER_CAP
    if total_abs_trade > max_abs_trade and total_abs_trade > 0:
        scale = max_abs_trade / total_abs_trade
        for t in decision_trades:
            scaled_delta = t["deltaDollar"] * scale
            price = next((p["price"] for p in positions if p["ticker"] == t["ticker"]), 0.0)
            current_shares = t["currentShares"]
            suggested_dollar_target = t["currentDollarValue"] + scaled_delta
            suggested_shares = safe_div(suggested_dollar_target, price, current_shares) if price > 0 else current_shares
            share_change = suggested_shares - current_shares
            current_weight = safe_div(t["currentDollarValue"], total_value, 0.0)
            suggested_weight = safe_div(suggested_dollar_target, total_value, 0.0)
            weight_change = suggested_weight - current_weight

            t["deltaDollar"] = round(scaled_delta, 2)
            t["suggestedDollarTarget"] = round(suggested_dollar_target, 2)
            t["suggestedTargetWeight"] = round(suggested_weight, 4)
            t["suggestedShares"] = round_shares(suggested_shares)
            t["shareChange"] = round_shares(share_change)

            if abs(t["deltaDollar"]) >= total_value * MIN_TRADE_PCT:
                if abs(t["shareChange"]) >= MIN_SHARE_DELTA and (abs(t["deltaDollar"]) >= MIN_TRADE_DOLLARS or abs(weight_change) >= MIN_WEIGHT_CHANGE):
                    t["actionFinal"] = "BUY" if t["shareChange"] > 0 else "SELL"
                else:
                    t["actionFinal"] = "HOLD"
            else:
                t["actionFinal"] = "HOLD"

            conviction_score = float(latest_scores.get(t["ticker"], 0.0))
            raw_priorities[t["ticker"]] = abs(t["deltaDollar"]) * (1.0 + abs(conviction_score))

        portfolio_flags.append("turnover_capped")

    max_priority = max(raw_priorities.values()) if raw_priorities else 1.0
    if max_priority <= 0:
        max_priority = 1.0
    for t in decision_trades:
        t["priorityScore"] = round(raw_priorities.get(t["ticker"], 0.0) / max_priority, 3)

    decision_trades.sort(key=lambda x: x["priorityScore"], reverse=True)
    actionable_trades = [t for t in decision_trades if t["actionFinal"] != "HOLD"]

    if len(actionable_trades) >= 6:
        portfolio_flags.append("high_turnover_signal_load")
    if any(w > 0.35 for w in current_sector_weights.values()):
        portfolio_flags.append("sector_concentration_high")
    if any(d["weight"] > 0.25 for d in diagnostics):
        portfolio_flags.append("single_name_concentration_high")

    target_sector_weights: Dict[str, float] = {}
    all_sectors = set(current_sector_weights.keys())
    for row in suggested_targets:
        s = row["sector"]
        all_sectors.add(s)
        target_sector_weights[s] = target_sector_weights.get(s, 0.0) + row["suggestedTargetWeight"]

    returns = close[live_tickers].pct_change() if (not close.empty and live_tickers) else pd.DataFrame()
    avg_vol = annualized_vol_from_returns(returns.tail(VOL_LOOKBACK).stack()) if not returns.empty else 0.0
    top_buys = selected[:5]
    weakest = [d["ticker"] for d in diagnostics[-5:]]

    top_actions = actionable_trades[:4]
    summary_lines = []
    if not top_actions:
        if selected:
            summary_lines.append("No meaningful trades today. Current holdings already near target.")
        else:
            summary_lines.append("No qualifying momentum candidates. Stay mostly in cash.")
    else:
        for t in top_actions:
            if t["actionFinal"] == "BUY":
                summary_lines.append(f"BUY {t['ticker']}: +{t['shareChange']} shares (~${abs(t['deltaDollar']):.2f})")
            elif t["actionFinal"] == "SELL":
                summary_lines.append(f"SELL {t['ticker']}: {t['shareChange']} shares (~${abs(t['deltaDollar']):.2f})")

    return {
        "metrics": {
            "positionCount": len(positions),
            "totalMarketValue": round(total_value, 2),
            "avgVol": round(avg_vol, 4),
            "buyCount": 0,
            "sellCount": 0,
            "actionableTradeCount": len(actionable_trades),
            "topBuys": ", ".join(top_buys),
            "weakestNames": ", ".join(weakest),
            "portfolioFlags": ", ".join(dict.fromkeys(portfolio_flags)) if portfolio_flags else "none",
            "selectedCount": len(selected),
            "topN": TOP_N,
            "momentumLookback": MOMENTUM_LOOKBACK,
            "volLookback": VOL_LOOKBACK,
            "trendLookback": TREND_LOOKBACK,
            "targetInvestedWeight": round(target_invested_weight, 4),
            "riskOn": risk_on,
        },
        "diagnostics": diagnostics,
        "suggestedTargets": suggested_targets,
        "sectorWeights": [
            {
                "sector": s,
                "currentWeight": round(current_sector_weights.get(s, 0.0), 4),
                "targetWeight": round(target_sector_weights.get(s, 0.0), 4),
            }
            for s in sorted(all_sectors, key=lambda x: target_sector_weights.get(x, 0.0), reverse=True)
        ],
        "decisionTrades": decision_trades,
        "emailSummary": {
            "generatedAt": snapshot.generatedAt,
            "topActions": top_actions,
            "summaryLines": summary_lines,
        },
    }
