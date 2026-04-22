#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLI", "XLY",
    "SMH", "SOXX", "NVDA", "AMD", "MSFT", "AAPL",
    "AMZN", "META", "AVGO", "TSLA", "COIN", "WGMI", "SOXL"
]
BENCHMARKS = ["SPY", "QQQ"]
LEVERAGED = {"SOXL", "TQQQ", "SPXL", "TECL", "UPRO", "FNGU"}

START = "2020-01-01"
OUTPUT_DIR = Path("sweep_output_quick")
TOP_NS = [2, 3, 4]
MOMENTUMS = [60, 90]
VOLS = [20]
TREND = 20
REGIME_MA = 200
TRANSACTION_COST_BPS = 5.0
LEVERAGED_CAP = 0.15
RISK_OFF_INVESTED_WEIGHT = 0.35


@dataclass
class RunConfig:
    top_n: int
    momentum: int
    vol: int
    use_regime_filter: bool
    benchmark_relative_filter: bool
    leveraged_mode: str


def download_close(symbols: Iterable[str], start: str) -> pd.DataFrame:
    symbols = list(dict.fromkeys([s.strip().upper() for s in symbols if s.strip()]))
    data = yf.download(
        tickers=symbols,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if data.empty:
        raise ValueError("No data downloaded")

    if isinstance(data.columns, pd.MultiIndex):
        frames = []
        for sym in symbols:
            if (sym, "Close") in data.columns:
                frames.append(data[(sym, "Close")].rename(sym))
        close = pd.concat(frames, axis=1)
    else:
        close = data[["Close"]].rename(columns={"Close": symbols[0]})

    return close.sort_index().ffill().dropna(how="all").astype(float)


def annualized_return(equity: pd.Series) -> float:
    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def annualized_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    return float(returns.std(ddof=0) * math.sqrt(252))


def sharpe_ratio(returns: pd.Series) -> float:
    vol = annualized_volatility(returns)
    if not np.isfinite(vol) or vol <= 1e-12:
        return np.nan
    return float((returns.mean() * 252) / vol)


def summarize_equity(equity: pd.Series) -> dict[str, float]:
    returns = equity.pct_change().dropna()
    return {
        "CAGR": annualized_return(equity),
        "MaxDrawdown": max_drawdown(equity),
        "Sharpe": sharpe_ratio(returns),
        "AnnualVol": annualized_volatility(returns),
        "TotalReturn": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
    }


def turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    idx = sorted(set(prev_w.index).union(new_w.index))
    a = prev_w.reindex(idx).fillna(0.0)
    b = new_w.reindex(idx).fillna(0.0)
    return float((a - b).abs().sum())


def run_strategy(close: pd.DataFrame, cfg: RunConfig) -> tuple[pd.Series, pd.DataFrame]:
    px = close.copy()
    rets = px.pct_change()

    momentum = px / px.shift(cfg.momentum) - 1.0
    vol = rets.rolling(cfg.vol).std(ddof=0)
    trend = px / px.shift(TREND) - 1.0
    score = momentum / vol.replace(0, np.nan)

    if cfg.benchmark_relative_filter and "SPY" in px.columns:
        spy_mom = momentum["SPY"]
        for c in px.columns:
            if c != "SPY":
                score[c] = score[c].where(momentum[c] > spy_mom)

    score = score.where(trend > 0)

    regime_on = pd.Series(True, index=px.index)
    if cfg.use_regime_filter:
        if "SPY" in px.columns:
            regime_on &= px["SPY"] > px["SPY"].rolling(REGIME_MA).mean()
        if "QQQ" in px.columns:
            regime_on &= px["QQQ"] > px["QQQ"].rolling(REGIME_MA).mean()

    shifted_score = score.shift(1)
    weights = pd.DataFrame(0.0, index=px.index, columns=px.columns, dtype=float)

    for dt in px.index:
        row = shifted_score.loc[dt].dropna().sort_values(ascending=False)
        selected = row.head(cfg.top_n).index.tolist()

        invest_weight = 1.0
        if cfg.use_regime_filter and not bool(regime_on.loc[dt]):
            invest_weight = RISK_OFF_INVESTED_WEIGHT

        if len(selected) < cfg.top_n:
            invest_weight *= len(selected) / max(cfg.top_n, 1)

        if not selected or invest_weight <= 0:
            continue

        base = invest_weight / len(selected)
        row_weights = {t: base for t in selected}

        if cfg.leveraged_mode == "off":
            for t in list(row_weights.keys()):
                if t in LEVERAGED:
                    row_weights[t] = 0.0
        elif cfg.leveraged_mode == "capped":
            for t in list(row_weights.keys()):
                if t in LEVERAGED:
                    row_weights[t] = min(row_weights[t], LEVERAGED_CAP)

        s = sum(row_weights.values())
        if s > 0:
            scale = invest_weight / s
            row_weights = {k: v * scale for k, v in row_weights.items()}

        for t, w in row_weights.items():
            weights.at[dt, t] = float(w)

    gross = (weights.shift(1).fillna(0.0) * rets.fillna(0.0)).sum(axis=1)

    prev = pd.Series(0.0, index=weights.columns, dtype=float)
    costs = []
    for dt in weights.index:
        new = weights.loc[dt].astype(float)
        t = turnover(prev, new)
        costs.append(t * (TRANSACTION_COST_BPS / 10000.0))
        prev = new

    cost_s = pd.Series(costs, index=weights.index)
    net = gross - cost_s
    equity = (1.0 + net).cumprod().rename("strategy")

    diag = pd.DataFrame({
        "gross": gross,
        "cost": cost_s,
        "net": net,
        "selected_count": (weights > 0).sum(axis=1),
        "invested_weight": weights.sum(axis=1),
        "regime_on": regime_on.astype(int),
    }, index=weights.index)
    return equity, diag


def benchmark_equity(close: pd.DataFrame, ticker: str) -> pd.Series:
    s = close[ticker].dropna()
    r = s.pct_change().fillna(0.0)
    return (1.0 + r).cumprod().rename(ticker)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    close = download_close(UNIVERSE, START)

    min_needed = max(max(MOMENTUMS), max(VOLS), TREND, REGIME_MA) + 5
    valid_cols = [c for c in close.columns if close[c].dropna().shape[0] >= min_needed]
    close = close[valid_cols].copy()

    bench = {}
    for b in BENCHMARKS:
        if b in close.columns:
            beq = benchmark_equity(close, b)
            stats = summarize_equity(beq)
            stats["Name"] = b
            bench[b] = stats

    configs = [
        RunConfig(top_n, mom, vol, regime, rel, lev)
        for top_n, mom, vol, regime, rel, lev in product(
            TOP_NS, MOMENTUMS, VOLS, [False, True], [False, True], ["off", "capped"]
        )
    ]

    runs = []
    for i, cfg in enumerate(configs, start=1):
        print(f"Running {i}/{len(configs)}: {cfg}")
        equity, diag = run_strategy(close, cfg)
        stats = summarize_equity(equity)
        row = {
            "top_n": cfg.top_n,
            "momentum": cfg.momentum,
            "vol": cfg.vol,
            "regime_filter": cfg.use_regime_filter,
            "benchmark_relative": cfg.benchmark_relative_filter,
            "leveraged_mode": cfg.leveraged_mode,
            **stats,
        }
        if "SPY" in bench:
            row["CAGR_vs_SPY"] = row["CAGR"] - bench["SPY"]["CAGR"]
            row["Sharpe_vs_SPY"] = row["Sharpe"] - bench["SPY"]["Sharpe"]
        if "QQQ" in bench:
            row["CAGR_vs_QQQ"] = row["CAGR"] - bench["QQQ"]["CAGR"]
            row["Sharpe_vs_QQQ"] = row["Sharpe"] - bench["QQQ"]["Sharpe"]

        composite = (
            (row["Sharpe"] * 3.0) +
            (row["CAGR"] * 2.0) +
            row.get("Sharpe_vs_SPY", 0.0) +
            row.get("Sharpe_vs_QQQ", 0.0) -
            abs(row["MaxDrawdown"]) * 1.2
        )
        row["CompositeScore"] = composite
        runs.append(row)

    runs_df = pd.DataFrame(runs).sort_values("CompositeScore", ascending=False)
    runs_df.to_csv(OUTPUT_DIR / "quick_sweep_summary.csv", index=False)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    print("\n=== TOP 15 RUNS ===")
    cols = [
        "top_n", "momentum", "vol", "regime_filter", "benchmark_relative",
        "leveraged_mode", "CAGR", "MaxDrawdown", "Sharpe",
        "CAGR_vs_SPY", "CAGR_vs_QQQ", "Sharpe_vs_SPY", "Sharpe_vs_QQQ", "CompositeScore"
    ]
    print(runs_df[cols].head(15).to_string(index=False))

    if bench:
        print("\n=== BENCHMARKS ===")
        print(pd.DataFrame(bench.values()).to_string(index=False))

    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
