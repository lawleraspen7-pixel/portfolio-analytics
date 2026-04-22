from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

SECRET = os.environ["SECRET"]

app = FastAPI()


class Position(BaseModel):
    row: Optional[int]
    ticker: str
    execute: str
    own: float
    price: float
    targetShares: float
    costBasis: float
    purchaseDate: str
    sector: str
    vol: float


class Snapshot(BaseModel):
    generatedAt: str
    positions: List[Position]


@app.get("/health")
def health():
    return {"ok": True}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-12 else default


def round_shares(x: float) -> float:
    return round(x, 3)


@app.post("/analyze")
def analyze(snapshot: Snapshot, x_analytics_secret: str = Header(default="")):
    if x_analytics_secret != SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    positions = [p for p in snapshot.positions if p.ticker and p.ticker != "CASH"]

    total_value = sum(p.own * p.price for p in positions)
    buy_count = sum(1 for p in positions if p.execute.upper().startswith("BUY"))
    sell_count = sum(1 for p in positions if p.execute.upper().startswith("SELL"))

    vols = [p.vol for p in positions if p.vol > 0]
    avg_vol = sum(vols) / len(vols) if vols else 0.0

    # -----------------------------
    # Sector concentration
    # -----------------------------
    sector_values = {}
    for p in positions:
        value = p.own * p.price
        sector = p.sector.strip() if p.sector else "Unknown"
        sector_values[sector] = sector_values.get(sector, 0.0) + value

    sector_weights = {
        sector: safe_div(value, total_value, 0.0)
        for sector, value in sector_values.items()
    }

    diagnostics = []
    raw_positive_scores = {}

    # -----------------------------
    # Score each position
    # -----------------------------
    for p in positions:
        value = p.own * p.price
        weight = safe_div(value, total_value, 0.0)
        gap = p.targetShares - p.own
        gap_value = gap * p.price

        is_buy = p.execute.upper().startswith("BUY")
        is_sell = p.execute.upper().startswith("SELL")
        is_hold = not is_buy and not is_sell

        sector = p.sector.strip() if p.sector else "Unknown"
        sector_weight = sector_weights.get(sector, 0.0)
        risk_score = clamp(p.vol, 0.0, 1.5)

        unrealized_pct = safe_div((p.price - p.costBasis), p.costBasis, 0.0) if p.costBasis > 0 else 0.0

        signal_score = 0.0
        if is_buy:
            signal_score += 1.0
        elif is_sell:
            signal_score -= 1.0

        vol_penalty = p.vol * 0.55
        weight_penalty = max(0.0, weight - 0.12) * 3.5
        sector_penalty = max(0.0, sector_weight - 0.30) * 2.0
        gap_strength = clamp(abs(gap_value) / max(total_value, 1.0), 0.0, 0.20) * 6.0

        if is_buy:
            signal_score += 0.55
            signal_score -= vol_penalty
            signal_score -= weight_penalty
            signal_score -= sector_penalty
            signal_score += gap_strength

            if p.vol < 0.35:
                signal_score += 0.20
            if p.vol > 0.80:
                signal_score -= 0.35
            if weight > 0.20:
                signal_score -= 0.40

        elif is_sell:
            signal_score -= 0.55
            signal_score += weight_penalty
            signal_score += sector_penalty
            signal_score += gap_strength

            if unrealized_pct < -0.08:
                signal_score -= 0.15
            if p.vol > 0.80:
                signal_score -= 0.20
            if weight > 0.20:
                signal_score -= 0.20

        elif is_hold:
            signal_score -= vol_penalty * 0.25
            signal_score -= weight_penalty * 0.25
            signal_score -= sector_penalty * 0.20

            if unrealized_pct > 0.10 and p.vol < 0.40:
                signal_score += 0.20
            if unrealized_pct < -0.15 and p.vol > 0.70:
                signal_score -= 0.20

        signal_score = clamp(signal_score, -3.0, 3.0)
        conviction_score = clamp(signal_score / 3.0, -1.0, 1.0)

        # Positive-only pool for target allocation
        positive_alloc_score = max(0.0, conviction_score)
        raw_positive_scores[p.ticker] = positive_alloc_score

        comments = []
        if abs(gap) > 0:
            comments.append(f"share_gap={gap:.3f}")
        if abs(gap_value) > 0:
            comments.append(f"gap_value={gap_value:.2f}")
        if weight > 0.20:
            comments.append("oversized")
        if sector_weight > 0.35:
            comments.append("crowded_sector")
        if p.vol > 0.80:
            comments.append("high_vol")
        elif p.vol < 0.30:
            comments.append("lower_vol")
        if unrealized_pct < -0.10:
            comments.append("deep_loser")
        elif unrealized_pct > 0.15:
            comments.append("strong_winner")

        diagnostics.append({
            "row": p.row,
            "ticker": p.ticker,
            "signalScore": round(signal_score, 4),
            "convictionScore": round(conviction_score, 4),
            "riskScore": round(risk_score, 4),
            "weight": round(weight, 4),
            "sectorWeight": round(sector_weight, 4),
            "currentDollarValue": round(value, 2),
            "unrealizedPct": round(unrealized_pct, 4),
            "gapShares": round(gap, 4),
            "gapValue": round(gap_value, 2),
            "comment": ", ".join(comments)
        })

    # -----------------------------
    # Suggested target weights
    # -----------------------------
    positive_total = sum(raw_positive_scores.values())

    suggested_targets = []
    for d in diagnostics:
        ticker = d["ticker"]
        alloc_score = raw_positive_scores.get(ticker, 0.0)

        suggested_weight = 0.0
        if positive_total > 0 and alloc_score > 0:
            suggested_weight = (alloc_score / positive_total) * 0.85

        # Per-position hard cap
        suggested_weight = min(suggested_weight, 0.20)

        suggested_targets.append({
            "ticker": ticker,
            "suggestedTargetWeight": round(suggested_weight, 4),
            "convictionScore": d["convictionScore"]
        })

    capped_sum = sum(x["suggestedTargetWeight"] for x in suggested_targets)
    if capped_sum > 0.85:
        scale = 0.85 / capped_sum
        for x in suggested_targets:
            x["suggestedTargetWeight"] = round(x["suggestedTargetWeight"] * scale, 4)

    target_weight_map = {x["ticker"]: x["suggestedTargetWeight"] for x in suggested_targets}

    # -----------------------------
    # Turn targets into trade decisions
    # -----------------------------
    decision_trades = []

    MIN_TRADE_DOLLARS = 25.0
    MIN_SHARE_DELTA = 0.10

    for p in positions:
        ticker = p.ticker
        price = p.price
        current_value = p.own * price
        suggested_weight = target_weight_map.get(ticker, 0.0)
        suggested_dollar_target = suggested_weight * total_value
        delta_dollar = suggested_dollar_target - current_value
        suggested_shares = safe_div(suggested_dollar_target, price, 0.0) if price > 0 else 0.0
        share_change = suggested_shares - p.own

        action_final = "HOLD"
        if abs(delta_dollar) >= MIN_TRADE_DOLLARS and abs(share_change) >= MIN_SHARE_DELTA:
            if share_change > 0:
                action_final = "BUY"
            elif share_change < 0:
                action_final = "SELL"

        conviction_score = next(
            (d["convictionScore"] for d in diagnostics if d["ticker"] == ticker),
            0.0
        )

        priority_score = abs(delta_dollar) * (1.0 + abs(conviction_score))

        decision_trades.append({
            "ticker": ticker,
            "currentShares": round(p.own, 4),
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

    # -----------------------------
    # Portfolio flags
    # -----------------------------
    portfolio_flags = []
    if avg_vol > 0.65:
        portfolio_flags.append("portfolio_vol_high")
    if any(w > 0.35 for w in sector_weights.values()):
        portfolio_flags.append("sector_concentration_high")
    if any(d["weight"] > 0.25 for d in diagnostics):
        portfolio_flags.append("single_name_concentration_high")

    # -----------------------------
    # Sort helper outputs
    # -----------------------------
    diagnostics.sort(key=lambda d: d["convictionScore"], reverse=True)
    suggested_targets.sort(key=lambda d: d["suggestedTargetWeight"], reverse=True)

    top_buys = [d["ticker"] for d in diagnostics if d["convictionScore"] > 0][:5]
    weakest = [d["ticker"] for d in sorted(diagnostics, key=lambda d: d["convictionScore"])][:5]

    # -----------------------------
    # Email summary block
    # -----------------------------
    top_actions = actionable_trades[:8]

    summary_lines = []
    if not top_actions:
        summary_lines.append("No meaningful trades today. Hold current positions.")
    else:
        for t in top_actions:
            if t["actionFinal"] == "BUY":
                summary_lines.append(
                    f"BUY {t['ticker']}: +{t['shareChange']} shares (~${t['deltaDollar']})"
                )
            elif t["actionFinal"] == "SELL":
                summary_lines.append(
                    f"SELL {t['ticker']}: {t['shareChange']} shares (~${abs(t['deltaDollar'])})"
                )

    return {
        "metrics": {
            "positionCount": len(positions),
            "totalMarketValue": round(total_value, 2),
            "avgVol": round(avg_vol, 4),
            "buyCount": buy_count,
            "sellCount": sell_count,
            "topBuys": ", ".join(top_buys),
            "weakestNames": ", ".join(weakest),
            "actionableTradeCount": len(actionable_trades),
            "portfolioFlags": ", ".join(portfolio_flags) if portfolio_flags else "none"
        },
        "diagnostics": diagnostics,
        "suggestedTargets": suggested_targets,
        "sectorWeights": [
            {"sector": k, "weight": round(v, 4)}
            for k, v in sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
        ],
        "decisionTrades": decision_trades,
        "emailSummary": {
            "generatedAt": snapshot.generatedAt,
            "topActions": top_actions,
            "summaryLines": summary_lines
        }
    }
