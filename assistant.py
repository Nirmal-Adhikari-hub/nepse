"""
NEPSE Signals assistant — a personalized chatbot for the website.

Two tiers, both FREE:
  • LLM mode  — if a free API key is present (GROQ_API_KEY preferred, else
    HF_TOKEN), it talks like ChatGPT/Claude, grounded in our live model data.
  • Grounded fallback — with no key, a deterministic responder still answers
    from the real model predictions + site facts (so it always works for $0).

The assistant is always GROUNDED: the relevant stock's live model numbers and
the site's honest stats are injected into the prompt/answer, so it represents
*this* website, not generic chatter.
"""
import os
import requests


SYSTEM = (
    "You are the assistant for 'NEPSE Signals', a website that gives honest, "
    "machine-learning DIRECTION forecasts for Nepal Stock Exchange (NEPSE) stocks. "
    "Persona: friendly, concise, plain-English first with optional detail; honest about "
    "uncertainty; never hypes. House rules you must follow: (1) This is educational, NOT "
    "financial advice — remind users when they ask what to buy. (2) The model predicts "
    "DIRECTION (up/down probability), not exact prices. (3) On HIGH-CONVICTION calls (top-20% confidence) "
    "it's right ~62% of the time; overall ~54% — modest but real, not random. Low-conviction signals are "
    "near a coin flip — the edge is in the confident picks. Say this honestly. (4) Use ONLY the "
    "GROUNDING DATA provided for any stock numbers; if a stock isn't in the data, say you don't "
    "have it. (5) Keep answers short unless asked to elaborate. (6) Never invent prices or guarantees."
)


def get_provider():
    """Return (base_url, key, model) for the first available free provider, or None."""
    g = os.environ.get("GROQ_API_KEY")
    if g:
        return ("https://api.groq.com/openai/v1/chat/completions", g, "llama-3.3-70b-versatile")
    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if hf:
        return ("https://router.huggingface.co/v1/chat/completions", hf,
                "meta-llama/Llama-3.1-8B-Instruct")
    return None


def llm_reply(messages, provider):
    base, key, model = provider
    r = requests.post(base, headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "messages": messages, "temperature": 0.3,
                            "max_tokens": 600}, timeout=40)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
def find_tickers(text, symbols):
    up = text.upper()
    return [s for s in symbols if s in up.replace("?", " ").replace(",", " ").split()
            or f" {s} " in f" {up} "]


def stock_facts(sym, scores, asof, horizons):
    r = scores[scores.symbol == sym]
    if not len(r):
        return f"{sym}: not in our dataset."
    r = r.iloc[0]
    parts = [f"{sym} (data as of {asof.get(sym,'?')}):"]
    for h in horizons:
        p = float(r[f"p{h}"]) * 100
        lean = "up" if p >= 50 else "down"
        parts.append(f"  {h}-day P(up)={p:.0f}% (leans {lean}, conviction {p-50:+.0f}pts)")
    if "vol10" in r and r["vol10"] == r["vol10"]:
        parts.append(f"  forecast 10-day volatility (risk) = {float(r['vol10']):.1f}%")
    return "\n".join(parts)


def grounding(query, scores, meta):
    horizons = meta["horizons"]; asof = meta.get("asof", {})
    m10 = meta["metrics"].get("10", meta["metrics"].get(10, {}))
    facts = [
        f"SITE FACTS: universe={meta['n_stocks']} NEPSE stocks ({meta.get('n_fresh','?')} current), "
        f"{meta['yrs']} yrs data, as of {meta['as_of']}. Model=global LightGBM, out-of-sample tested. "
        f"10-day directional accuracy={m10.get('acc')}% (baseline {m10.get('maj')}%, edge "
        f"+{m10.get('edge')}pts), high-conviction accuracy~{m10.get('acc20')}%, overfitting PBO="
        f"{m10.get('pbo')} (≈0 good). Predicts DIRECTION not price. Not financial advice."]
    for s in find_tickers(query, list(scores.symbol))[:4]:
        facts.append(stock_facts(s, scores, asof, horizons))
    return "\n".join(facts)


def rule_based(query, scores, meta):
    """Deterministic grounded answer when no LLM key is set."""
    q = query.lower(); horizons = meta["horizons"]; asof = meta.get("asof", {})
    m10 = meta["metrics"].get("10", meta["metrics"].get(10, {}))
    tickers = find_tickers(query, list(scores.symbol))
    if tickers:
        out = []
        for s in tickers[:3]:
            out.append(stock_facts(s, scores, asof, horizons))
        out.append("\n_Direction forecast, not a price target. ~55% accurate — not advice._")
        return "\n".join(out)
    if any(w in q for w in ["accura", "how good", "trust", "reliable", "work"]):
        return (f"On 10-day direction the model is right about **{m10.get('acc')}%** of the time — "
                f"a real **+{m10.get('edge')} pts** over the naive baseline ({m10.get('maj')}%), and up to "
                f"**{m10.get('acc20')}%** on its most confident calls. Overfitting check PBO={m10.get('pbo')} "
                f"(≈0 = trustworthy). It's modest but real — wrong ~45% of the time. Not financial advice.")
    if any(w in q for w in ["buy", "recommend", "invest", "pick", "should i"]):
        return ("Use the **Home → Get your personalized recommendation** tool: set your risk appetite, "
                "horizon and budget and it builds a basket from the model's conviction + risk. "
                "Remember: educational only, the model is wrong ~45% of the time — never invest what you "
                "can't afford to lose.")
    if any(w in q for w in ["how", "model", "method", "feature"]):
        return ("A global LightGBM model studies ~23 years across the whole NEPSE universe — momentum, "
                "volatility, RSI, 52-week range, volume and overall-market regime — and predicts each "
                "stock's up/down probability over 5/10/20 days. It's tested out-of-sample with an "
                "overfitting check. See **Model & Evidence** for the interactive charts.")
    return ("I'm the NEPSE Signals assistant. Ask me about a stock (e.g. 'How does NABIL look?'), how "
            "accurate the model is, how it works, or how to get a recommendation. Educational only — not "
            "financial advice.")


def narrative(facts_text, provider=None):
    """LLM investment perspective for one stock. Returns text, or None if no LLM."""
    provider = provider or get_provider()
    if provider is None:
        return None
    msgs = [
        {"role": "system", "content": SYSTEM + " TASK: write a concise (~150 word) INVESTMENT "
         "PERSPECTIVE for ONE stock in 2 short paragraphs: (1) what the model's signals say — momentum, "
         "trend, strength, risk, and the bull-vs-bear balance; (2) a bottom-line stance for conservative "
         "vs aggressive investors. Plain English, balanced, honest about the ~55% accuracy. End with a "
         "one-line italic '_Not financial advice._'. Use ONLY the data provided; never invent numbers."},
        {"role": "user", "content": facts_text}]
    try:
        return llm_reply(msgs, provider)
    except Exception:
        return None


def answer(query, history, scores, meta):
    provider = get_provider()
    ctx = grounding(query, scores, meta)
    if provider is None:
        return rule_based(query, scores, meta), "grounded"
    msgs = [{"role": "system", "content": SYSTEM + "\n\nGROUNDING DATA (use only this for numbers):\n" + ctx}]
    for role, content in history[-6:]:
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": query})
    try:
        return llm_reply(msgs, provider), "llm"
    except Exception:
        return rule_based(query, scores, meta), "grounded"
