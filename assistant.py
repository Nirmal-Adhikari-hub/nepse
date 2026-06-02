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
    "Persona: warm, friendly financial explainer for EVERYDAY people. ALWAYS lead with a one-line plain "
    "verdict (e.g. '🟢 NABIL looks mildly positive') and a clear BOTTOM-LINE takeaway. Weave numbers in "
    "lightly and in plain words ('about 6 in 10 odds'); NEVER dump raw probability lists. End with one short "
    "optional technical line for nerds. Honest about uncertainty; never hypes. House rules: (1) This is educational, NOT "
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


def _lean(p):
    c = (p - .5) * 100
    if c >= 6:  return "clearly up", "🟢"
    if c >= 1.5: return "slightly up", "🟢"
    if c > -1.5: return "flat / uncertain", "🟡"
    if c > -6:  return "slightly down", "🔴"
    return "clearly down", "🔴"


def friendly_stock(sym, scores, meta):
    """Plain-language, recommendation-style summary of one stock (no jargon up front)."""
    r = scores[scores.symbol == sym]
    if not len(r):
        return None
    r = r.iloc[0]; asof = meta.get("asof", {}).get(sym, meta["as_of"])
    p5, p10, p20 = float(r["p5"]), float(r["p10"]), float(r["p20"])
    lm, em = _lean(p10); ll, _ = _lean(p20)
    vol = float(r["vol10"]) if r["vol10"] == r["vol10"] else None
    riskw = "low" if (vol or 6) < 5 else "high" if (vol or 6) > 9 else "moderate"
    avg = (p5 + p10 + p20) / 3
    if avg >= .53:   head = f"🟢 **{sym} looks broadly positive** right now"
    elif avg <= .47: head = f"🔴 **{sym} looks weak** right now"
    else:            head = f"🟡 **{sym} is a mixed bag** right now"
    if p10 >= .53 and (vol or 6) < 9:
        bottom = "Could be worth a look for a 1–2 week move — keep the position small and confirm with your own view."
    elif p20 >= .53 and p10 < .5:
        bottom = "Soft in the short run but improving over a month — more of a *watch* than a buy-now."
    elif avg <= .47:
        bottom = "The model isn't keen here right now — likely one to **avoid or wait** on."
    else:
        bottom = "No strong signal either way — better to wait for a clearer setup."
    stale = "" if asof >= "2026-01-01" else f" _(based on data to {asof})_"
    return (f"{head}{stale}\n\n"
            f"- **Next 1–2 weeks:** {lm} {em}\n"
            f"- **Next ~month:** {ll}\n"
            f"- **Risk:** {riskw}{f' (≈{vol:.0f}% typical swing)' if vol else ''}\n\n"
            f"**Bottom line:** {bottom}\n\n"
            f"_Educational only, not financial advice — the model is right ~6 times in 10 on its confident calls._\n\n"
            f"*🔎 the numbers — chance of rising: 1wk {p5*100:.0f}% · 2wk {p10*100:.0f}% · 1mo {p20*100:.0f}%*")


ACCURACY_PLAIN = (
    "**The simple version:** imagine the model makes **100 predictions**. 🎯\n\n"
    "- About **57 turn out right**, 43 wrong — a little better than flipping a coin (which is 50/50).\n"
    "- For the stocks it's **most sure about**, about **64–65 out of 100** are right.\n"
    "- For the **overall market's** up/down direction, about **56 out of 100** (up to ~68 on its surest weeks).\n\n"
    "So when we say *“57% accurate”* we mean **“right 57 times out of every 100 calls”** — **not** “sure 57% of "
    "the time.” It's a real but small edge: enough to **tilt the odds in your favour** across many picks over time, "
    "but never a guarantee on any single stock. (Markets are brutally hard — even professionals rarely beat ~60% honestly.)")


def explain_metric(metric, value_text, sym, provider=None):
    """LLM micro-explanation of one indicator's current reading for one stock. None if no LLM."""
    provider = provider or get_provider()
    if provider is None:
        return None
    msgs = [{"role": "system", "content":
             "You explain ONE stock indicator to a beginner in 2 short, friendly sentences: (1) what it measures "
             "in plain words, (2) what THIS specific reading suggests for the stock. No jargon dumps, honest, never "
             "give a buy/sell order, no price targets."},
            {"role": "user", "content": f"Stock: {sym}. Indicator: {metric}. Current reading: {value_text}. "
             f"Explain simply what it means for {sym} right now."}]
    try:
        return llm_reply(msgs, provider)
    except Exception:
        return None


def rule_based(query, scores, meta):
    """Friendly, plain-language grounded answer when no LLM key is set."""
    q = query.lower()
    tickers = find_tickers(query, list(scores.symbol))
    if tickers:
        out = [friendly_stock(s, scores, meta) for s in tickers[:2]]
        out = [o for o in out if o]
        if out:
            return "\n\n---\n\n".join(out)
        return f"I don't have data on that one. Try a ticker like NABIL, NMB, or ADBL."
    if any(w in q for w in ["accura", "how good", "trust", "reliable", "right", "correct", "%", "percent"]):
        return ACCURACY_PLAIN
    if any(w in q for w in ["buy", "recommend", "invest", "pick", "should i", "watch", "top"]):
        return ("Easiest way: open **Home → 🎯 Get your personalized recommendation**, pick your risk level, "
                "horizon and budget, and it builds you a basket of the strongest stocks (and warns you if the "
                "overall market looks weak). Or just ask me *“how does NABIL look?”* for any stock.\n\n"
                "_Educational only — never invest money you can't afford to lose._")
    if any(w in q for w in ["how", "model", "method", "feature", "what is"]):
        return ("In plain terms: the model has studied **23 years** of every NEPSE stock — how prices have been "
                "moving, how choppy they are, whether they're near highs or lows, trading volume, and the overall "
                "market mood — and from that it estimates each stock's **odds of rising** over the next week to a "
                "month. It was tested only on data it had never seen, so the accuracy is honest. See **Model & "
                "Evidence** for the proof.")
    return ("Hi! I can help you make sense of NEPSE in plain English 🙂 Try:\n\n"
            "- *“How does NABIL look?”* — a friendly read on any stock\n"
            "- *“How accurate is this?”* — what the numbers really mean\n"
            "- *“What should I buy?”* — how to get a recommendation\n\n"
            "_Educational only — not financial advice._")


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
