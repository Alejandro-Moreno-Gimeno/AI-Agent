import numpy as np
import pandas as pd


def generate_order_book_snapshot(mid: float = 100.0, n_levels: int = 5) -> pd.DataFrame:
    """
    Generate a synthetic limit order book snapshot around a given mid price.

    Parameters
    ----------
    mid : float
        Mid price around which to create bid/ask levels.
    n_levels : int
        Number of price levels on each side.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: side ('bid'/'ask'), price, size.
    """
    prices_up = mid + np.arange(1, n_levels + 1) * 0.1
    prices_down = mid - np.arange(1, n_levels + 1) * 0.1

    bid_prices = prices_down
    ask_prices = prices_up

    bid_sizes = np.random.randint(10, 50, size=n_levels)
    ask_sizes = np.random.randint(10, 50, size=n_levels)

    book = pd.DataFrame({
        "side": ["bid"] * n_levels + ["ask"] * n_levels,
        "price": np.concatenate([bid_prices, ask_prices]),
        "size": np.concatenate([bid_sizes, ask_sizes])
    })

    return book
