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

@app.post("/analyze")
def analyze(snapshot: Snapshot, x_analytics_secret: str = Header(default="")):
    if x_analytics_secret != SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    positions = [p for p in snapshot.positions if p.ticker and p.ticker != "CASH"]

    total_value = sum(p.own * p.price for p in positions)
    avg_vol = sum(p.vol for p in positions if p.vol > 0) / max(1, sum(1 for p in positions if p.vol > 0))

    diagnostics = []
    for p in positions:
        value = p.own * p.price
        weight = value / total_value if total_value > 0 else 0
        gap = p.targetShares - p.own

        signal_score = 0.0
        if p.execute.startswith("BUY"):
            signal_score = 1.0
        elif p.execute.startswith("SELL"):
            signal_score = -1.0

        risk_score = min(1.0, p.vol) if p.vol > 0 else 0.0

        comments = []
        if abs(gap) > 0:
            comments.append(f"share_gap={gap:.3f}")
        if weight > 0.25:
            comments.append("large_weight")
        if p.vol > 0.8:
            comments.append("high_vol")

        diagnostics.append({
            "ticker": p.ticker,
            "signalScore": round(signal_score, 4),
            "riskScore": round(risk_score, 4),
            "comment": ", ".join(comments)
        })

    return {
        "metrics": {
            "positionCount": len(positions),
            "totalMarketValue": round(total_value, 2),
            "avgVol": round(avg_vol, 4),
            "buyCount": sum(1 for p in positions if p.execute.startswith("BUY")),
            "sellCount": sum(1 for p in positions if p.execute.startswith("SELL"))
        },
        "diagnostics": diagnostics
    }
