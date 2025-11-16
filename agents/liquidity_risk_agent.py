from typing import Sequence
from openai import OpenAI

from src.config import get_openai_api_key
from src.microstructure import assess_liquidity_risk


def _get_client() -> OpenAI:
    api_key = get_openai_api_key()
    return OpenAI(api_key=api_key)


def generate_liquidity_risk_alert(
    spreads: Sequence[float],
    liq_scores: Sequence[float],
    regimes: Sequence[str],
    lookback: int = 200,
) -> str:
    """
    Use microstructure summary + an LLM to generate a risk-focused alert.
    """

    client = _get_client()

    risk_info = assess_liquidity_risk(spreads, liq_scores, regimes, lookback=lookback)

    prompt = f"""
You are a risk-focused fixed income assistant.

Based on the following aggregate liquidity indicators for the last {risk_info["lookback"]} observations:

- Average spread: {risk_info["avg_spread"]:.4f}
- Average liquidity score: {risk_info["avg_liq_score"]:.2f}
- Fraction of time in 'stressed' regime: {risk_info["frac_stressed"]:.2%}
- Fraction of time in 'high_liquidity' regime: {risk_info["frac_high_liq"]:.2%}
- Overall risk level (preliminary rule-based classification): {risk_info["risk_level"]!r}

Write a short (3–5 sentences) risk alert for a trader or risk manager.

Guidelines:
- Focus ONLY on liquidity and execution risk, not price direction.
- If risk_level is 'high', be explicit about potential difficulties in executing size and possible slippage.
- If 'medium', mention moderate concerns but that conditions are still manageable.
- If 'low', mention that liquidity conditions are broadly supportive.
- Do NOT give any investment recommendation, only describe risk conditions.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a cautious, risk-focused assistant for fixed income markets."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()
