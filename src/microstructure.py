import pandas as pd
from typing import Dict


def compute_book_metrics(book_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute core microstructure metrics from a limit order book snapshot.

    Parameters
    ----------
    book_df : pd.DataFrame
        DataFrame with columns: side ('bid'/'ask'), price, size.

    Returns
    -------
    dict
        Dictionary with best_bid, best_ask, spread, mid, bid_depth,
        ask_depth, total_depth, imbalance.
    """
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
    """
    Convenience function returning (spread, mid, imbalance) tuple — useful for time series loops.
    """
    metrics = compute_book_metrics(book_df)
    return metrics["spread"], metrics["mid"], metrics["imbalance"]
