#!/usr/bin/env python3
"""
Momentum backtest for a small ticker universe.

What it does:
- Downloads daily prices with yfinance
- Computes:
    * momentum = 90-day return
    * volatility = 20-day stddev of daily returns
    * score = momentum / volatility
    * optional 20-day trend filter (exclude if 20-day return <= 0)
- Rebalances daily into top N tickers by score
- Compares strategy vs SPY and QQQ
- Writes CSV outputs

Usage:
    pip install yfinance pandas numpy
    python momentum_backtest.py

Optional:
    python momentum_backtest.py --tickers RKLB NVDA QQQ AVGO AMZN SOXL SMH
    python momentum_backtest.py --start 2021-01-01 --top-n 5 --momentum-lookback 90 --vol-lookback 20
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = [
    "RKLB", "NVDA", "SMH", "SOXL", "AMZN", "QQQ", "AVGO"
]
DEFAULT_BENCHMARKS = ["SPY", "QQQ"]


@dataclass
class Config:
    tickers: list[str]
    benchmarks: list[str]
    start: str
    end: str | None
    top_n: int
    momentum_lookback: int
    vol_lookback: int
    trend_lookback: int
    exclude_nonpositive_trend: bool
    output_dir: str
    transaction_cost_bps: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--benchmarks", nargs="*", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--momentum-lookback", type=int, default=90)
    parser.add_argument("--vol-lookback", type=int, default=20)
    parser.add_argument("--trend-lookback", type=int, default=20)
    parser.add_argument("--exclude-nonpositive-trend", action="store_true", default=True)
    parser.add_argument("--allow-nonpositive-trend", dest="exclude_nonpositive_trend", action="store_false")
    parser.add_argument("--output-dir", default="backtest_output")
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0,
                        help="Applied on notional turnover each rebalance. Example: 5 = 5 bps")
    args = parser.parse_args()

    tickers = [t.upper().strip() for t in args.tickers if t.strip()]
    benchmarks = [t.upper().strip() for t in args.benchmarks if t.strip()]
    tickers = list(dict.fromkeys(tickers))
    benchmarks = list(dict.fromkeys(benchmarks))

    if not tickers:
        raise ValueError("No tickers provided.")
    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1")
    if args.top_n > len(tickers):
        raise ValueError("--top-n cannot exceed number of tickers")

    return Config(
        tickers=tickers,
        benchmarks=benchmarks,
        start=args.start,
        end=args.end,
        top_n=args.top_n,
        momentum_lookback=args.momentum_lookback,
        vol_lookback=args.vol_lookback,
        trend_lookback=args.trend_lookback,
        exclude_nonpositive_trend=args.exclude_nonpositive_trend,
        output_dir=args.output_dir,
        transaction_cost_bps=args.transaction_cost_bps,
    )


def download_close_prices(symbols: Iterable[str], start: str, end: str | None) -> pd.DataFrame:
    symbols = list(dict.fromkeys([s.upper().strip() for s in symbols if s.strip()]))
    if not symbols:
        raise ValueError("No symbols to download.")

    data = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if data.empty:
        raise ValueError("No data downloaded. Check tickers/date range/internet connection.")

    # yfinance shape differs for 1 vs many symbols
    if isinstance(data.columns, pd.MultiIndex):
        close_frames = []
        for sym in symbols:
            if sym in data.columns.get_level_values(0):
                if (sym, "Close") in data.columns:
                    s = data[(sym, "Close")].rename(sym)
                    close_frames.append(s)
        if not close_frames:
            raise ValueError("Could not find Close columns in downloaded data.")
        close = pd.concat(close_frames, axis=1)
    else:
        if "Close" not in data.columns:
            raise ValueError("Could not find Close column in downloaded data.")
        close = data[["Close"]].rename(columns={"Close": symbols[0]})

    close = close.sort_index().ffill().dropna(how="all")
    return close


def annualized_return(equity: pd.Series) -> float:
    if equity.empty or len(equity) < 2:
        return np.nan
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    if start_val <= 0 or end_val <= 0:
        return np.nan
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return np.nan
    years = days / 365.25
    return (end_val / start_val) ** (1 / years) - 1


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def annualized_volatility(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    return float(returns.std(ddof=0) * math.sqrt(252))


def sharpe_ratio(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    vol = annualized_volatility(returns)
    if not np.isfinite(vol) or vol <= 1e-12:
        return np.nan
    return float((returns.mean() * 252) / vol)


def turnover_from_weights(prev_weights: pd.Series, new_weights: pd.Series) -> float:
    idx = sorted(set(prev_weights.index).union(new_weights.index))
    prev_aligned = prev_weights.reindex(idx).fillna(0.0)
    new_aligned = new_weights.reindex(idx).fillna(0.0)
    return float((new_aligned - prev_aligned).abs().sum())


def build_strategy(
    close: pd.DataFrame,
    tickers: list[str],
    top_n: int,
    momentum_lookback: int,
    vol_lookback: int,
    trend_lookback: int,
    exclude_nonpositive_trend: bool,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    close = close[tickers].copy()
    returns = close.pct_change()

    momentum = close / close.shift(momentum_lookback) - 1.0
    vol = returns.rolling(vol_lookback).std(ddof=0)
    trend = close / close.shift(trend_lookback) - 1.0

    score = momentum / vol.replace(0, np.nan)

    if exclude_nonpositive_trend:
        score = score.where(trend > 0)

    # Use yesterday's signal for today's holdings to avoid same-day lookahead
    signal = score.shift(1)

    weights = pd.DataFrame(0.0, index=close.index, columns=tickers)

    for dt in weights.index:
        row = signal.loc[dt].dropna().sort_values(ascending=False)
        chosen = row.head(top_n).index.tolist()
        if chosen:
            weights.loc[dt, chosen] = 1.0 / len(chosen)

    asset_returns = returns.fillna(0.0)
    gross_strategy_returns = (weights.shift(1).fillna(0.0) * asset_returns).sum(axis=1)

    # Turnover cost applied on rebalance dates
    prev_w = pd.Series(0.0, index=tickers)
    costs = []
    for dt in weights.index:
        new_w = weights.loc[dt]
        turnover = turnover_from_weights(prev_w, new_w)
        cost = turnover * (transaction_cost_bps / 10000.0)
        costs.append(cost)
        prev_w = new_w

    cost_series = pd.Series(costs, index=weights.index, name="cost")
    net_strategy_returns = gross_strategy_returns - cost_series
    equity = (1.0 + net_strategy_returns).cumprod().rename("strategy")

    diagnostics = pd.DataFrame({
        "gross_return": gross_strategy_returns,
        "cost": cost_series,
        "net_return": net_strategy_returns,
        "selected_count": (weights > 0).sum(axis=1),
    }, index=weights.index)

    return weights, equity, diagnostics


def build_benchmark_equity(close: pd.DataFrame, benchmark: str) -> pd.Series:
    bench_close = close[benchmark].dropna()
    bench_returns = bench_close.pct_change().fillna(0.0)
    return (1.0 + bench_returns).cumprod().rename(benchmark)


def summarize_equity(equity: pd.Series) -> dict[str, float]:
    returns = equity.pct_change().dropna()
    return {
        "CAGR": annualized_return(equity),
        "MaxDrawdown": max_drawdown(equity),
        "Sharpe": sharpe_ratio(returns),
        "AnnualVol": annualized_volatility(returns),
        "TotalReturn": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) >= 2 else np.nan,
    }


def main() -> None:
    cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_symbols = list(dict.fromkeys(cfg.tickers + cfg.benchmarks))
    close = download_close_prices(all_symbols, cfg.start, cfg.end)

    # Keep only symbols with enough data
    min_history = max(cfg.momentum_lookback, cfg.vol_lookback, cfg.trend_lookback) + 5
    valid_cols = [c for c in close.columns if close[c].dropna().shape[0] >= min_history]
    close = close[valid_cols].copy()

    tickers = [t for t in cfg.tickers if t in close.columns]
    benchmarks = [b for b in cfg.benchmarks if b in close.columns]

    if len(tickers) < cfg.top_n:
        raise ValueError(f"Not enough valid ticker histories. Valid tickers: {tickers}")

    weights, strategy_equity, diagnostics = build_strategy(
        close=close,
        tickers=tickers,
        top_n=cfg.top_n,
        momentum_lookback=cfg.momentum_lookback,
        vol_lookback=cfg.vol_lookback,
        trend_lookback=cfg.trend_lookback,
        exclude_nonpositive_trend=cfg.exclude_nonpositive_trend,
        transaction_cost_bps=cfg.transaction_cost_bps,
    )

    summary_rows = []
    strategy_summary = summarize_equity(strategy_equity)
    strategy_summary["Name"] = "STRATEGY"
    summary_rows.append(strategy_summary)

    combined_equity = pd.DataFrame({"STRATEGY": strategy_equity})

    for bench in benchmarks:
        bench_equity = build_benchmark_equity(close, bench)
        combined_equity[bench] = bench_equity.reindex(combined_equity.index).ffill()
        row = summarize_equity(bench_equity)
        row["Name"] = bench
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)[["Name", "CAGR", "MaxDrawdown", "Sharpe", "AnnualVol", "TotalReturn"]]
    summary = summary.sort_values("Name").reset_index(drop=True)

    latest_weights = weights.iloc[-1]
    latest_selected = latest_weights[latest_weights > 0].sort_values(ascending=False)
    latest_selected_df = latest_selected.rename("weight").reset_index().rename(columns={"index": "ticker"})

    # Save files
    close.to_csv(output_dir / "prices_close.csv")
    weights.to_csv(output_dir / "strategy_weights.csv")
    diagnostics.to_csv(output_dir / "strategy_diagnostics.csv")
    combined_equity.to_csv(output_dir / "equity_curves.csv")
    summary.to_csv(output_dir / "summary.csv", index=False)
    latest_selected_df.to_csv(output_dir / "latest_selection.csv", index=False)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    print("\n=== CONFIG ===")
    print(cfg)

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== LATEST SELECTION ===")
    if latest_selected_df.empty:
        print("No positions selected on last day.")
    else:
        print(latest_selected_df.to_string(index=False))

    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
