from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import math
import pandas as pd
import numpy as np
import yfinance as yf

SECRET = os.environ["SECRET"]

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
# Strategy config
# =========================
TOP_N = 3
MOMENTUM_LOOKBACK = 90
VOL_LOOKBACK = 20
TREND_LOOKBACK = 20

TARGET_INVESTED_WEIGHT = 1.0
MIN_TRADE_DOLLARS = 300.0
MIN_SHARE_DELTA = 0.10
MIN_WEIGHT_CHANGE = 0.03

DOWNLOAD_PERIOD = "9mo"


# =========================
# Helpers
# =========================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-12 else default


def round_shares(x: float) -> float:
    return round(x, 3)


def clean_ticker(x: str) -> str:
    return (x or "").strip().upper()


def clean_execute(x: str) -> str:
    return (x or "").strip().upper()


def clean_sector(x: str) -> str:
    x = (x or "").strip()
    return x if x else "Unknown"


def annualized_vol_from_returns(returns: pd.Series) -> float:
    if returns is None or len(returns.dropna()) == 0:
        return 0.0
    return float(returns.std(ddof=0) * math.sqrt(252))


def build_price_frame(tickers: List[str]) -> pd.DataFrame:
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

    close = close.sort_index().ffill().dropna(how="all")
    return close


def compute_momentum_model(close: pd.DataFrame) -> Dict[str, Any]:
    if close.empty:
        return {
            "latest_scores": {},
            "latest_momentum": {},
            "latest_trend": {},
            "latest_vol": {},
            "selected": [],
            "valid_tickers": []
        }

    returns = close.pct_change()
    momentum = close / close.shift(MOMENTUM_LOOKBACK) - 1.0
    vol = returns.rolling(VOL_LOOKBACK).std(ddof=0)
    trend = close / close.shift(TREND_LOOKBACK) - 1.0

    score = momentum / vol.replace(0.0, np.nan)
    score = score.where(trend > 0)

    latest_score_row = score.iloc[-1].dropna().sort_values(ascending=False)
    latest_momentum_row = momentum.iloc[-1].to_dict()
    latest_trend_row = trend.iloc[-1].to_dict()
    latest_vol_row = vol.iloc[-1].to_dict()

    selected = latest_score_row.head(TOP_N).index.tolist()

    return {
        "latest_scores": {k: float(v) for k, v in latest_score_row.to_dict().items() if pd.notna(v)},
        "latest_momentum": {k: float(v) for k, v in latest_momentum_row.items() if pd.notna(v)},
        "latest_trend": {k: float(v) for k, v in latest_trend_row.items() if pd.notna(v)},
        "latest_vol": {k: float(v) for k, v in latest_vol_row.items() if pd.notna(v)},
        "selected": selected,
        "valid_tickers": list(close.columns)
    }


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
            "execute": clean_execute(raw.execute),
            "own": max(0.0, float(raw.own or 0.0)),
            "price": max(0.0, float(raw.price or 0.0)),
            "targetShares": max(0.0, float(raw.targetShares or 0.0)),
            "costBasis": max(0.0, float(raw.costBasis or 0.0)),
            "purchaseDate": raw.purchaseDate or "",
            "sector": clean_sector(raw.sector),
            "vol": max(0.0, float(raw.vol or 0.0)),
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

    tickers = [p["ticker"] for p in positions]
    buy_count = sum(1 for p in positions if p["execute"].startswith("BUY"))
    sell_count = sum(1 for p in positions if p["execute"].startswith("SELL"))

    close = build_price_frame(tickers)
    model = compute_momentum_model(close)

    latest_scores = model["latest_scores"]
    latest_momentum = model["latest_momentum"]
    latest_trend = model["latest_trend"]
    latest_vol = model["latest_vol"]
    selected = model["selected"]
    valid_tickers = set(model["valid_tickers"])

    # Sector map from current holdings
    sector_values: Dict[str, float] = {}
    for p in positions:
        value = p["own"] * p["price"]
        sector_values[p["sector"]] = sector_values.get(p["sector"], 0.0) + value

    current_sector_weights = {
        sector: safe_div(value, total_value, 0.0)
        for sector, value in sector_values.items()
    }

    diagnostics = []
    target_weights: Dict[str, float] = {}

    if selected:
        equal_weight = TARGET_INVESTED_WEIGHT / len(selected)
        for ticker in tickers:
            target_weights[ticker] = equal_weight if ticker in selected else 0.0
    else:
        for ticker in tickers:
            target_weights[ticker] = 0.0

    portfolio_flags = []
    if len(selected) < TOP_N:
        portfolio_flags.append("partial_cash_signal")
    if len(valid_tickers) < len(tickers):
        portfolio_flags.append("missing_market_data")
    if not selected:
        portfolio_flags.append("no_positive_trend_candidates")

    # Diagnostics
    for p in positions:
        ticker = p["ticker"]
        value = p["own"] * p["price"]
        weight = safe_div(value, total_value, 0.0)
        sector_weight = current_sector_weights.get(p["sector"], 0.0)
        unrealized_pct = safe_div((p["price"] - p["costBasis"]), p["costBasis"], 0.0) if p["costBasis"] > 0 else 0.0

        momentum_90d = latest_momentum.get(ticker)
        trend_20d = latest_trend.get(ticker)
        vol_20d = latest_vol.get(ticker)
        score = latest_scores.get(ticker)

        comments = []
        if ticker in selected:
            comments.append("selected_top_momentum")
        if ticker not in valid_tickers:
            comments.append("missing_price_history")
        if trend_20d is not None and trend_20d <= 0:
            comments.append("trend_filter_failed")
        if vol_20d is not None and vol_20d > 0.50:
            comments.append("higher_recent_vol")
        if weight > 0.20:
            comments.append("oversized")
        if sector_weight > 0.35:
            comments.append("crowded_sector")

        conviction_score = 0.0
        if score is not None:
            conviction_score = float(score)

        diagnostics.append({
            "row": p["row"],
            "ticker": ticker,
            "sector": p["sector"],
            "signalScore": round(conviction_score, 4),
            "convictionScore": round(conviction_score, 4),
            "riskScore": round(float(vol_20d) if vol_20d is not None else 0.0, 4),
            "weight": round(weight, 4),
            "sectorWeight": round(sector_weight, 4),
            "currentDollarValue": round(value, 2),
            "unrealizedPct": round(unrealized_pct, 4),
            "gapShares": round(target_weights.get(ticker, 0.0) * total_value / p["price"] - p["own"], 4) if p["price"] > 0 else 0.0,
            "gapValue": round(target_weights.get(ticker, 0.0) * total_value - value, 2),
            "comment": ", ".join(comments),
            "momentum90d": round(float(momentum_90d), 4) if momentum_90d is not None else None,
            "trend20d": round(float(trend_20d), 4) if trend_20d is not None else None,
            "vol20d": round(float(vol_20d), 4) if vol_20d is not None else None
        })

    diagnostics.sort(key=lambda d: (d["convictionScore"] if d["convictionScore"] is not None else -999999), reverse=True)

    suggested_targets = []
    for p in positions:
        ticker = p["ticker"]
        suggested_targets.append({
            "ticker": ticker,
            "sector": p["sector"],
            "suggestedTargetWeight": round(target_weights.get(ticker, 0.0), 4),
            "convictionScore": round(float(latest_scores.get(ticker, 0.0)), 4)
        })

    suggested_targets.sort(key=lambda d: d["suggestedTargetWeight"], reverse=True)
    target_weight_map = {x["ticker"]: x["suggestedTargetWeight"] for x in suggested_targets}

    decision_trades = []
    for p in positions:
        ticker = p["ticker"]
        price = p["price"]
        current_value = p["own"] * price
        current_weight = safe_div(current_value, total_value, 0.0)
        suggested_weight = target_weight_map.get(ticker, 0.0)
        suggested_dollar_target = suggested_weight * total_value
        delta_dollar = suggested_dollar_target - current_value
        suggested_shares = safe_div(suggested_dollar_target, price, 0.0) if price > 0 else 0.0
        share_change = suggested_shares - p["own"]
        weight_change = suggested_weight - current_weight

        action_final = "HOLD"
        if (
            (abs(delta_dollar) >= MIN_TRADE_DOLLARS or abs(weight_change) >= MIN_WEIGHT_CHANGE)
            and abs(share_change) >= MIN_SHARE_DELTA
        ):
            action_final = "BUY" if share_change > 0 else "SELL"

        conviction_score = float(latest_scores.get(ticker, 0.0))
        priority_score = abs(delta_dollar) * (1.0 + abs(conviction_score))

        decision_trades.append({
            "ticker": ticker,
            "sector": p["sector"],
            "currentShares": round(p["own"], 4),
            "currentDollarValue": round(current_value, 2),
            "suggestedTargetWeight": round(suggested_weight, 4),
            "suggestedDollarTarget": round(suggested_dollar_target, 2),
            "deltaDollar": round(delta_dollar, 2),
            "suggestedShares": round_shares(suggested_shares),
            "shareChange": round_shares(share_change),
            "actionFinal": action_final,
            "priorityScore": round(priority_score, 2)
        })

    decision_trades.sort(key=lambda x: x["priorityScore"], reverse=True)
    actionable_trades = [t for t in decision_trades if t["actionFinal"] != "HOLD"]

    if len(actionable_trades) >= 8:
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

    avg_vol = annualized_vol_from_returns(close.pct_change().tail(VOL_LOOKBACK).stack()) if not close.empty else 0.0
    top_buys = [ticker for ticker in selected][:5]
    weakest = [d["ticker"] for d in diagnostics[-5:]]

    top_actions = actionable_trades[:8]
    summary_lines = []
    if not top_actions:
        if selected:
            summary_lines.append("No meaningful trades today. Current holdings already near target.")
        else:
            summary_lines.append("No qualifying momentum candidates. Stay in cash / hold no-trade posture.")
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
            "buyCount": buy_count,
            "sellCount": sell_count,
            "actionableTradeCount": len(actionable_trades),
            "topBuys": ", ".join(top_buys),
            "weakestNames": ", ".join(weakest),
            "portfolioFlags": ", ".join(portfolio_flags) if portfolio_flags else "none",
            "selectedCount": len(selected),
            "topN": TOP_N,
            "momentumLookback": MOMENTUM_LOOKBACK,
            "volLookback": VOL_LOOKBACK,
            "trendLookback": TREND_LOOKBACK
        },
        "diagnostics": diagnostics,
        "suggestedTargets": suggested_targets,
        "sectorWeights": [
            {
                "sector": sector,
                "currentWeight": round(current_sector_weights.get(sector, 0.0), 4),
                "targetWeight": round(target_sector_weights.get(sector, 0.0), 4)
            }
            for sector in sorted(all_sectors, key=lambda s: target_sector_weights.get(s, 0.0), reverse=True)
        ],
        "decisionTrades": decision_trades,
        "emailSummary": {
            "generatedAt": snapshot.generatedAt,
            "topActions": top_actions,
            "summaryLines": summary_lines
        }
    }
