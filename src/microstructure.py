import pandas as pd
from typing import Dict


def compute_book_metrics(book_df: pd.DataFrame) -> Dict[str, float]:
    bids = book_df[book_df["side"] == "bid"].sort_values("price", ascending=False)
    asks = book_df[book_df["side"] == "ask"].sort_values("price", ascending=True)

    best_bid = bids["price"].max()
    best_ask = asks["price"].min()

    spread = best_ask - best_bid
    mid = (best_bid + best_ask) / 2

    bid_depth = bids["size"].sum()
    ask_depth = asks["size"].sum()
    total_depth = bid_depth + ask_depth

    imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "mid": mid,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "total_depth": total_depth,
        "imbalance": imbalance,
    }


def compute_metrics_from_snapshot(book_df: pd.DataFrame):
    metrics = compute_book_metrics(book_df)
    return metrics["spread"], metrics["mid"], metrics["imbalance"]


# ✅ ADD THESE TWO NEW FUNCTIONS
def compute_liquidity_score(spread: float, total_depth: float) -> float:
    """Higher = more liquid. Simple score combining depth + spread."""
    if spread <= 0:
        spread = 1e-6
    return float(total_depth / spread)


def classify_liquidity_regime(spread: float, imbalance: float, liquidity_score: float) -> str:
    if liquidity_score > 500 and abs(imbalance) < 0.2 and spread < 0.3:
        return "high_liquidity"

    if liquidity_score < 150 or spread > 0.6:
        return "stressed"

    if imbalance > 0.5:
        return "one_sided_buy"

    if imbalance < -0.5:
        return "one_sided_sell"

    return "normal"

from typing import Sequence


def assess_liquidity_risk(
    spreads: Sequence[float],
    liq_scores: Sequence[float],
    regimes: Sequence[str],
    lookback: int = 200,
) -> dict:
    """
    Compute some simple aggregate risk indicators over the last N points.
    """

    spreads_tail = spreads[-lookback:]
    liq_tail = liq_scores[-lookback:]
    regimes_tail = regimes[-lookback:]

    avg_spread = float(sum(spreads_tail) / len(spreads_tail))
    frac_stressed = regimes_tail.count("stressed") / len(regimes_tail)
    frac_high_liq = regimes_tail.count("high_liquidity") / len(regimes_tail)
    avg_liq_score = float(sum(liq_tail) / len(liq_tail))

    # Simple traffic-light logic (you can tune this)
    if frac_stressed > 0.4 or avg_spread > 0.8 or avg_liq_score < 150:
        level = "high"
    elif frac_stressed > 0.2 or avg_spread > 0.5 or avg_liq_score < 250:
        level = "medium"
    else:
        level = "low"

    return {
        "lookback": lookback,
        "avg_spread": avg_spread,
        "avg_liq_score": avg_liq_score,
        "frac_stressed": frac_stressed,
        "frac_high_liq": frac_high_liq,
        "risk_level": level,
    }
