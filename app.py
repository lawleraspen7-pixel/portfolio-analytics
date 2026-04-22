from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os

SECRET = os.environ["SECRET"]

app = FastAPI()


# =========================
# Models
# =========================
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
# Config
# =========================
TARGET_INVESTED_WEIGHT = 0.85
MAX_POSITION_WEIGHT = 0.20
MAX_SECTOR_WEIGHT = 0.35

REBALANCE_ALPHA = 0.25          # move 25% toward ideal per run
MAX_DAILY_TURNOVER = 0.25       # max abs trade dollars as % of portfolio per day

MIN_TRADE_DOLLARS = 50.0
MIN_SHARE_DELTA = 0.10

# soft behavior controls
SELL_BIAS = 0.20                # easier to reduce names explicitly marked SELL
KEEP_FLOOR_NON_SELL = 0.015     # don't zero tiny non-sell positions too aggressively
HIGH_VOL_CUTOFF = 0.80
LOW_VOL_CUTOFF = 0.30


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


def force_scale_to_sum(items: List[Dict[str, Any]], key: str, target_sum: float) -> None:
    current_sum = sum(max(0.0, float(item.get(key, 0.0))) for item in items)
    if current_sum <= 0:
        return
    scale = target_sum / current_sum
    for item in items:
        item[key] = max(0.0, float(item.get(key, 0.0)) * scale)


def apply_position_cap(items: List[Dict[str, Any]], key: str, cap: float) -> None:
    # iterative cap + redistribute among uncapped positive weights
    for _ in range(10):
        total = sum(max(0.0, float(item.get(key, 0.0))) for item in items)
        if total <= 0:
            return

        excess = 0.0
        uncapped_total = 0.0

        for item in items:
            w = max(0.0, float(item.get(key, 0.0)))
            if w > cap:
                excess += w - cap
                item[key] = cap
            else:
                uncapped_total += w

        if excess <= 1e-9:
            return

        if uncapped_total <= 1e-9:
            return

        for item in items:
            w = max(0.0, float(item.get(key, 0.0)))
            if w < cap:
                item[key] = w + excess * (w / uncapped_total)


def apply_sector_cap(items: List[Dict[str, Any]], key: str, sector_key: str, sector_cap: float) -> None:
    # iterative soft enforcement
    for _ in range(12):
        sector_totals: Dict[str, float] = {}
        total = 0.0
        for item in items:
            w = max(0.0, float(item.get(key, 0.0)))
            s = str(item.get(sector_key, "Unknown"))
            sector_totals[s] = sector_totals.get(s, 0.0) + w
            total += w

        if total <= 0:
            return

        over = {s: w for s, w in sector_totals.items() if w > sector_cap}
        if not over:
            return

        freed = 0.0
        eligible_total = 0.0

        for item in items:
            s = str(item.get(sector_key, "Unknown"))
            w = max(0.0, float(item.get(key, 0.0)))
            if s in over and sector_totals[s] > 0:
                ratio = sector_cap / sector_totals[s]
                new_w = w * ratio
                freed += w - new_w
                item[key] = new_w

        for item in items:
            s = str(item.get(sector_key, "Unknown"))
            if s not in over:
                eligible_total += max(0.0, float(item.get(key, 0.0)))

        if freed <= 1e-9 or eligible_total <= 1e-9:
            return

        for item in items:
            s = str(item.get(sector_key, "Unknown"))
            if s not in over:
                w = max(0.0, float(item.get(key, 0.0)))
                item[key] = w + freed * (w / eligible_total)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
def analyze(snapshot: Snapshot, x_analytics_secret: str = Header(default="")):
    if x_analytics_secret != SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    positions = []
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

    buy_count = sum(1 for p in positions if p["execute"].startswith("BUY"))
    sell_count = sum(1 for p in positions if p["execute"].startswith("SELL"))
    vols = [p["vol"] for p in positions if p["vol"] > 0]
    avg_vol = sum(vols) / len(vols) if vols else 0.0

    # current sector map
    sector_values: Dict[str, float] = {}
    for p in positions:
        value = p["own"] * p["price"]
        sector_values[p["sector"]] = sector_values.get(p["sector"], 0.0) + value

    current_sector_weights = {
        sector: safe_div(value, total_value, 0.0)
        for sector, value in sector_values.items()
    }

    diagnostics = []
    positive_alloc_scores: Dict[str, float] = {}

    # =========================
    # Score positions
    # =========================
    for p in positions:
        ticker = p["ticker"]
        value = p["own"] * p["price"]
        weight = safe_div(value, total_value, 0.0)
        gap_shares = p["targetShares"] - p["own"]
        gap_value = gap_shares * p["price"]
        sector_weight = current_sector_weights.get(p["sector"], 0.0)
        unrealized_pct = safe_div((p["price"] - p["costBasis"]), p["costBasis"], 0.0) if p["costBasis"] > 0 else 0.0

        is_buy = p["execute"].startswith("BUY")
        is_sell = p["execute"].startswith("SELL")
        is_hold = not is_buy and not is_sell

        # base preference from execute
        signal_score = 0.0
        if is_buy:
            signal_score += 1.00
        elif is_sell:
            signal_score -= 1.00

        # structure penalties / rewards
        vol_penalty = p["vol"] * 0.55
        size_penalty = max(0.0, weight - 0.12) * 3.20
        sector_penalty = max(0.0, sector_weight - 0.30) * 2.25
        gap_strength = clamp(abs(gap_value) / total_value, 0.0, 0.20) * 5.50

        if is_buy:
            signal_score += 0.50
            signal_score += gap_strength
            signal_score -= vol_penalty
            signal_score -= size_penalty
            signal_score -= sector_penalty

            if p["vol"] < LOW_VOL_CUTOFF:
                signal_score += 0.20
            if p["vol"] > HIGH_VOL_CUTOFF:
                signal_score -= 0.35
            if unrealized_pct > 0.10:
                signal_score += 0.10

        elif is_sell:
            signal_score -= 0.50
            signal_score -= SELL_BIAS
            signal_score += size_penalty
            signal_score += sector_penalty
            signal_score += gap_strength

            if p["vol"] > HIGH_VOL_CUTOFF:
                signal_score -= 0.20
            if unrealized_pct < -0.08:
                signal_score -= 0.15

        elif is_hold:
            signal_score -= vol_penalty * 0.25
            signal_score -= size_penalty * 0.25
            signal_score -= sector_penalty * 0.20

            if unrealized_pct > 0.10 and p["vol"] < 0.40:
                signal_score += 0.20
            if unrealized_pct < -0.15 and p["vol"] > 0.70:
                signal_score -= 0.20

        signal_score = clamp(signal_score, -3.0, 3.0)
        conviction_score = clamp(signal_score / 3.0, -1.0, 1.0)

        # positive pool for allocation
        alloc_score = max(0.0, conviction_score)
        if is_sell:
            alloc_score = 0.0
        positive_alloc_scores[ticker] = alloc_score

        comments = []
        if abs(gap_shares) > 0:
            comments.append(f"share_gap={gap_shares:.3f}")
        if abs(gap_value) > 0:
            comments.append(f"gap_value={gap_value:.2f}")
        if weight > 0.20:
            comments.append("oversized")
        if sector_weight > 0.35:
            comments.append("crowded_sector")
        if p["vol"] > HIGH_VOL_CUTOFF:
            comments.append("high_vol")
        elif p["vol"] < LOW_VOL_CUTOFF:
            comments.append("lower_vol")
        if unrealized_pct < -0.10:
            comments.append("deep_loser")
        elif unrealized_pct > 0.15:
            comments.append("strong_winner")
        if is_buy:
            comments.append("buy_signal")
        elif is_sell:
            comments.append("sell_signal")

        diagnostics.append({
            "row": p["row"],
            "ticker": ticker,
            "sector": p["sector"],
            "signalScore": round(signal_score, 4),
            "convictionScore": round(conviction_score, 4),
            "riskScore": round(clamp(p["vol"], 0.0, 1.5), 4),
            "weight": round(weight, 4),
            "sectorWeight": round(sector_weight, 4),
            "currentDollarValue": round(value, 2),
            "unrealizedPct": round(unrealized_pct, 4),
            "gapShares": round(gap_shares, 4),
            "gapValue": round(gap_value, 2),
            "comment": ", ".join(comments)
        })

    conviction_map = {d["ticker"]: d["convictionScore"] for d in diagnostics}
    positive_total = sum(positive_alloc_scores.values())

    # =========================
    # Build ideal target weights
    # =========================
    target_rows = []
    for p in positions:
        ticker = p["ticker"]
        current_weight = safe_div(p["own"] * p["price"], total_value, 0.0)
        alloc_score = positive_alloc_scores.get(ticker, 0.0)

        ideal_weight = 0.0
        if positive_total > 0 and alloc_score > 0:
            ideal_weight = (alloc_score / positive_total) * TARGET_INVESTED_WEIGHT

        # Do not crush non-sell names straight to zero.
        if not p["execute"].startswith("SELL"):
            ideal_weight = max(ideal_weight, min(current_weight, KEEP_FLOOR_NON_SELL))

        target_rows.append({
            "ticker": ticker,
            "sector": p["sector"],
            "currentWeight": current_weight,
            "idealWeight": ideal_weight,
            "convictionScore": conviction_map.get(ticker, 0.0)
        })

    # Enforce structure on ideal portfolio
    force_scale_to_sum(target_rows, "idealWeight", TARGET_INVESTED_WEIGHT)
    apply_position_cap(target_rows, "idealWeight", MAX_POSITION_WEIGHT)
    force_scale_to_sum(target_rows, "idealWeight", TARGET_INVESTED_WEIGHT)
    apply_sector_cap(target_rows, "idealWeight", "sector", MAX_SECTOR_WEIGHT)
    force_scale_to_sum(target_rows, "idealWeight", TARGET_INVESTED_WEIGHT)

    # Blend from current -> ideal
    suggested_targets = []
    for row in target_rows:
        suggested_weight = (
            row["currentWeight"] * (1.0 - REBALANCE_ALPHA) +
            row["idealWeight"] * REBALANCE_ALPHA
        )
        suggested_targets.append({
            "ticker": row["ticker"],
            "sector": row["sector"],
            "suggestedTargetWeight": suggested_weight,
            "convictionScore": row["convictionScore"]
        })

    # Re-apply structure after blending
    force_scale_to_sum(suggested_targets, "suggestedTargetWeight", TARGET_INVESTED_WEIGHT)
    apply_position_cap(suggested_targets, "suggestedTargetWeight", MAX_POSITION_WEIGHT)
    force_scale_to_sum(suggested_targets, "suggestedTargetWeight", TARGET_INVESTED_WEIGHT)
    apply_sector_cap(suggested_targets, "suggestedTargetWeight", "sector", MAX_SECTOR_WEIGHT)
    force_scale_to_sum(suggested_targets, "suggestedTargetWeight", TARGET_INVESTED_WEIGHT)

    for row in suggested_targets:
        row["suggestedTargetWeight"] = round(row["suggestedTargetWeight"], 4)

    target_weight_map = {x["ticker"]: x["suggestedTargetWeight"] for x in suggested_targets}

    # =========================
    # Turn targets into trade decisions
    # =========================
    decision_trades = []
    for p in positions:
        ticker = p["ticker"]
        price = p["price"]
        current_value = p["own"] * price
        suggested_weight = target_weight_map.get(ticker, 0.0)
        suggested_dollar_target = suggested_weight * total_value
        delta_dollar = suggested_dollar_target - current_value
        suggested_shares = safe_div(suggested_dollar_target, price, 0.0) if price > 0 else 0.0
        share_change = suggested_shares - p["own"]
        conviction_score = conviction_map.get(ticker, 0.0)

        action_final = "HOLD"
        if abs(delta_dollar) >= MIN_TRADE_DOLLARS and abs(share_change) >= MIN_SHARE_DELTA:
            action_final = "BUY" if share_change > 0 else "SELL"

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

    # Turnover cap
    total_abs_trade = sum(abs(t["deltaDollar"]) for t in decision_trades)
    max_abs_trade = total_value * MAX_DAILY_TURNOVER

    if total_abs_trade > max_abs_trade and total_abs_trade > 0:
        scale = max_abs_trade / total_abs_trade
        for t in decision_trades:
            scaled_delta = t["deltaDollar"] * scale
            price = next((p["price"] for p in positions if p["ticker"] == t["ticker"]), 0.0)
            current_shares = t["currentShares"]
            suggested_dollar_target = t["currentDollarValue"] + scaled_delta
            suggested_shares = safe_div(suggested_dollar_target, price, current_shares) if price > 0 else current_shares
            share_change = suggested_shares - current_shares

            t["deltaDollar"] = round(scaled_delta, 2)
            t["suggestedDollarTarget"] = round(suggested_dollar_target, 2)
            t["suggestedShares"] = round_shares(suggested_shares)
            t["shareChange"] = round_shares(share_change)

            if abs(t["deltaDollar"]) >= MIN_TRADE_DOLLARS and abs(t["shareChange"]) >= MIN_SHARE_DELTA:
                t["actionFinal"] = "BUY" if t["shareChange"] > 0 else "SELL"
            else:
                t["actionFinal"] = "HOLD"

            conviction_score = conviction_map.get(t["ticker"], 0.0)
            t["priorityScore"] = round(abs(t["deltaDollar"]) * (1.0 + abs(conviction_score)), 2)

    decision_trades.sort(key=lambda x: x["priorityScore"], reverse=True)
    actionable_trades = [t for t in decision_trades if t["actionFinal"] != "HOLD"]

    # target sector weights
    target_sector_weights: Dict[str, float] = {}
    for row in suggested_targets:
        target_sector_weights[row["sector"]] = target_sector_weights.get(row["sector"], 0.0) + row["suggestedTargetWeight"]

    portfolio_flags = []
    if avg_vol > 0.65:
        portfolio_flags.append("portfolio_vol_high")
    if any(w > 0.35 for w in current_sector_weights.values()):
        portfolio_flags.append("sector_concentration_high")
    if any(d["weight"] > 0.25 for d in diagnostics):
        portfolio_flags.append("single_name_concentration_high")
    if len(actionable_trades) >= 8:
        portfolio_flags.append("high_turnover_signal_load")

    diagnostics.sort(key=lambda d: d["convictionScore"], reverse=True)
    suggested_targets.sort(key=lambda d: d["suggestedTargetWeight"], reverse=True)

    top_buys = [d["ticker"] for d in diagnostics if d["convictionScore"] > 0][:5]
    weakest = [d["ticker"] for d in sorted(diagnostics, key=lambda d: d["convictionScore"])][:5]

    top_actions = actionable_trades[:8]
    summary_lines = []
    if not top_actions:
        summary_lines.append("No meaningful trades today. Hold current positions.")
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
            "targetInvestedWeight": round(TARGET_INVESTED_WEIGHT, 4),
            "maxPositionWeight": round(MAX_POSITION_WEIGHT, 4),
            "maxSectorWeight": round(MAX_SECTOR_WEIGHT, 4),
            "rebalanceAlpha": round(REBALANCE_ALPHA, 4),
            "maxDailyTurnover": round(MAX_DAILY_TURNOVER, 4)
        },
        "diagnostics": diagnostics,
        "suggestedTargets": [
            {
                "ticker": x["ticker"],
                "sector": x["sector"],
                "suggestedTargetWeight": x["suggestedTargetWeight"],
                "convictionScore": x["convictionScore"]
            }
            for x in suggested_targets
        ],
        "sectorWeights": [
            {
                "sector": sector,
                "currentWeight": round(current_sector_weights.get(sector, 0.0), 4),
                "targetWeight": round(weight, 4)
            }
            for sector, weight in sorted(target_sector_weights.items(), key=lambda kv: kv[1], reverse=True)
        ],
        "decisionTrades": decision_trades,
        "emailSummary": {
            "generatedAt": snapshot.generatedAt,
            "topActions": top_actions,
            "summaryLines": summary_lines
        }
    }
