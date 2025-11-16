import os
import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --- Make sure we can import from src/ and agents/ ---

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_generation import generate_order_book_snapshot
from src.microstructure import (
    compute_book_metrics,
    compute_liquidity_score,
    classify_liquidity_regime,
)
from src.microstructure import assess_liquidity_risk
from agents.liquidity_reporter import generate_liquidity_commentary
from agents.liquidity_risk_agent import generate_liquidity_risk_alert


def simulate_time_series(
    n_steps: int = 1000,
    mid_start: float = 100.0,
    drift_sigma: float = 0.02,
    n_levels: int = 5,
) -> dict:
    """
    Run a simple microstructure simulation and return time series arrays.
    """
    spreads: List[float] = []
    mids: List[float] = []
    imbalances: List[float] = []
    liq_scores: List[float] = []
    regimes: List[str] = []

    np.random.seed(0)
    mid = mid_start

    for _ in range(n_steps):
        # Small random walk in mid price
        mid += np.random.normal(0, drift_sigma)
        book = generate_order_book_snapshot(mid=mid, n_levels=n_levels)

        metrics = compute_book_metrics(book)
        spread = metrics["spread"]
        midprice = metrics["mid"]
        imbalance = metrics["imbalance"]
        total_depth = metrics["total_depth"]

        liq_score = compute_liquidity_score(spread, total_depth)
        regime = classify_liquidity_regime(spread, imbalance, liq_score)

        spreads.append(spread)
        mids.append(midprice)
        imbalances.append(imbalance)
        liq_scores.append(liq_score)
        regimes.append(regime)

    return {
        "spreads": spreads,
        "mids": mids,
        "imbalances": imbalances,
        "liq_scores": liq_scores,
        "regimes": regimes,
    }


def main():
    st.set_page_config(page_title="AI Liquidity Assistant", layout="wide")

    st.title("🧠 AI Liquidity Assistant (Synthetic Microstructure Demo)")
    st.write(
        "This demo simulates a synthetic order book, computes microstructure metrics, "
        "classifies liquidity regimes, and uses AI agents to generate commentary and risk alerts."
    )

    # --- Sidebar controls ---
    st.sidebar.header("Simulation Settings")
    n_steps = st.sidebar.slider("Number of time steps", min_value=200, max_value=2000, value=1000, step=100)
    mid_start = st.sidebar.number_input("Initial mid price", value=100.0)
    drift_sigma = st.sidebar.slider("Mid price volatility (sigma)", min_value=0.005, max_value=0.1, value=0.02)
    n_levels = st.sidebar.slider("Order book levels per side", min_value=3, max_value=10, value=5)
    lookback = st.sidebar.slider("Lookback for AI analysis", min_value=50, max_value=500, value=200, step=50)

    if st.sidebar.button("Run Simulation"):
        with st.spinner("Simulating order book and computing metrics..."):
            results = simulate_time_series(
                n_steps=n_steps,
                mid_start=mid_start,
                drift_sigma=drift_sigma,
                n_levels=n_levels,
            )

        spreads = results["spreads"]
        mids = results["mids"]
        imbalances = results["imbalances"]
        liq_scores = results["liq_scores"]
        regimes = results["regimes"]

        # --- Plot section ---
        st.subheader("Microstructure Metrics Over Time")

        df = pd.DataFrame({
            "Midprice": mids,
            "Spread": spreads,
            "Liquidity Score": liq_scores,
            "Imbalance": imbalances,
        })

        # Midprice and spread (two columns)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Midprice**")
            st.line_chart(df["Midprice"])

        with col2:
            st.markdown("**Bid-Ask Spread**")
            st.line_chart(df["Spread"])

        # Liquidity score
        st.markdown("**Liquidity Score**")
        st.line_chart(df["Liquidity Score"])

        # Regimes as a colored strip
        st.markdown("**Liquidity Regimes**")
        regime_colors = {
            "high_liquidity": "green",
            "normal": "blue",
            "stressed": "red",
            "one_sided_buy": "orange",
            "one_sided_sell": "purple",
        }
        color_series = [regime_colors[r] for r in regimes]

        fig, ax = plt.subplots(figsize=(12, 1.5))
        ax.scatter(range(len(regimes)), [0] * len(regimes), c=color_series, marker="s", s=10)
        ax.set_yticks([])
        ax.set_xlabel("Time")
        ax.set_title("Liquidity Regimes")
        st.pyplot(fig)

        # --- AI agents section ---
        st.subheader("AI-Generated Commentary & Risk Alerts")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("**Liquidity Commentary**")
            try:
                comment = generate_liquidity_commentary(
                    spreads=spreads,
                    liq_scores=liq_scores,
                    imbalances=imbalances,
                    regimes=regimes,
                    lookback=lookback,
                )
                st.write(comment)
            except Exception as e:
                st.error(f"Error generating commentary: {e}")

        with colB:
            st.markdown("**Liquidity Risk Alert**")
            try:
                alert = generate_liquidity_risk_alert(
                    spreads=spreads,
                    liq_scores=liq_scores,
                    regimes=regimes,
                    lookback=lookback,
                )
                st.write(alert)
            except Exception as e:
                st.error(f"Error generating risk alert: {e}")

        # Optional: show raw risk metrics
        st.subheader("Aggregate Risk Indicators")
        risk_info = assess_liquidity_risk(spreads, liq_scores, regimes, lookback=lookback)
        st.json(risk_info)

    else:
        st.info("Use the controls in the sidebar and click **Run Simulation** to begin.")


if __name__ == "__main__":
    main()
