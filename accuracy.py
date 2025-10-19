import json, os, random, re

SEED = 42
random.seed(SEED)
# Fixes seed so demos are deterministic

DATA_PATH = os.path.join(os.path.dirname(__file__),"Headlines.jsonl")
# Dataset path defaults to Data/Headlines.jsonl, except if overriden with env var HEADLINES_PATH
# os.getenv() returns the value of the os environment variable key if it exists, otherwise returns the default value

HEDGES = {"may","might","could","reportedly","allegedly","rumor","sources say"}
CLICKBAIT = {"BREAKING","SHOCKING","MUST SEE","INCREDIBLE","!!!"}
WHITELIST = {"reuters","ap","associated press","ft","financial times","bloomberg","ons","imf","oecd","ecb","boe","sec"}
BLACKLIST = {"tiktok","randomblog","whatsapp","telegram","meme","clicknews","facebook","forum"}
# Dictionary of surface cues that will be used to assess confidence level (Hedges = Weak Language; Less Reliable, Clickbait = Extreme Language ; Lower Confidence, Whitelist/Blacklish = Source Analyser ; Nudges Confidence Up/Down)

# How to combine ML confidence (from dataset) with runtime heuristics
USE_ML_ONLY = True      # use ONLY the baked ML confidence for known ids
ML_BLEND = 0.8          # if USE_ML_ONLY=False, blend = ML_BLEND*ML + (1-ML_BLEND)*heuristic

class AccuracyEngine:
    def __init__(self, path=DATA_PATH):
        self.items = []
        # Open the JSONL file (one JSON object per line)
        with open(path, "r", encoding="utf-8") as f:  # utf-8 handles symbols like £, accents, etc
            for line in f:
                line = line.strip()  # Remove whitespace and newline
                if not line:
                    continue         # Skip completely blank lines
                self.items.append(json.loads(line))  # Parse JSON text: Python dict and store

        # Build a fast lookup: ID returns full record
        self.idx = {}
        for it in self.items:
            self.idx[it["id"]] = it

    def get_headline(self, domain=None):
        # Build a pool filtered by domain (or everything if domain is None)
        pool = []
        for item in self.items:
            if (domain is None) or (item.get("domain") == domain):
                pool.append(item)

        if not pool:
            # Clear error if the pool is empty (eg unknown domain)
            raise ValueError(f"No headlines available for domain={domain!r}")

        # Pick one at random to show the player (truth/confidence are not exposed here)
        choice = random.choice(pool)
        return {
            "headline_id": choice["id"],
            "text": choice["text"],
            "domain": choice["domain"],
            "source": choice.get("source", ""), # .get avoids KeyError if missing
            "url": choice.get("url", ""),
            "article_preview": choice.get("article_preview", "")
        }

    def _heuristic_conf(self, text, source, domain):
        # Start from neutral and adjust up/down based on cues
        score = 0.5
        reasons = []  # Collect human-readable reasons for the final rationale

        s_low = (source or "").lower()  # Normalise source for case-insensitive checks
        t = (text or "").strip()        # Guard against None and stray spaces
        # Only keep A–Z and '!' to make detection simple and fast
        t_upper = re.sub(r"[^A-Z!]", "", t.upper())

        # Source Prior: Strong positive/negative nudge if we recognise the outlet
        for w in WHITELIST:
            if w in s_low:
                score += 0.35
                reasons.append("Trusted source")
                break
        for w in BLACKLIST:
            if w in s_low:
                score -= 0.35
                reasons.append("Unreliable source")
                break

        # Hedging Language: Each match subtracts a bit, Capped so it doesn’t dominate
        hedge_hits = 0
        for w in HEDGES:
            # \b = word boundary; re.escape handles phrases like "sources say"
            if re.search(rf"\b{re.escape(w)}\b", t.lower()):
                hedge_hits += 1
        if hedge_hits > 0:
            score -= min(0.2, 0.05 * hedge_hits)  # cap total penalty at 0.2
            reasons.append("Hedging language")

        # Sensational Phrasing: All-caps tokens or “!!!” = downweight
        found_clickbait = False
        for w in CLICKBAIT:
            if w in t_upper:
                found_clickbait = True
                break
        if found_clickbait or ("!!!" in t):
            score -= 0.15
            reasons.append("Sensational phrasing")

        # Verifiability Cues: Numbers/Years/Percentages act as strong statistic
        if re.search(r"\b\d{4}\b", t) or re.search(r"\b\d+(\.\d+)?%?\b", t):
            score += 0.07
            reasons.append("Quantitative claim")

        # Institutions: Regulator/Agency acronyms add a little confidence
        if re.search(r"\b(SEC|FOMC|ECB|BoE|ONS|IMF)\b", t):
            score += 0.08
            reasons.append("Institution cited")

        # Domain Priors: Social content is noisier on average; News/Econ often refer to official reports
        if domain == "social":
            score -= 0.1
            reasons.append("Social media source")

        if (domain in {"news", "econ"}) and re.search(r"\b(report|release|statement|transcript)\b", t.lower()):
            score += 0.05
            reasons.append("Refers to official report")

        # Keep score within [0.01, 0.99] so Nothing is Absolute
        if score < 0.01:
            score = 0.01
        if score > 0.99:
            score = 0.99

        return score, reasons  # Numeric Confidence Scale + Which Rules Fired

    def _mk_rationale(self, item, reasons):
        parts = []

        # If the item is labeled true and comes from a trusted outlet, Explain that
        if item and item.get("truth"):
            src = (item.get("source", "") or "").lower()
            for w in WHITELIST:
                if w in src:
                    parts.append("Cross-verified by trusted outlets.")
                    break

        # Translate “reasons” into phrases
        if "Institution cited" in reasons:
            parts.append("Official/primary source referenced.")
        if "Hedging language" in reasons:
            parts.append("Heavy hedging without corroboration.")
        if "Sensational phrasing" in reasons:
            parts.append("Sensational tone indicates lower reliability.")
        if "Social media source" in reasons:
            parts.append("Originates from social media; needs cross-check.")

        # Fallbacks: Use dataset rationale if present, Otherwise a generic line
        if (not parts) and item:
            parts.append(item.get("rationale", "Confidence derived from linguistic + source cues."))
        if not parts:
            parts.append("Confidence derived from linguistic + source cues.")

        return " ".join(parts)

    def evaluate(self, headline_id=None, text=None):
    # Caller must pass either a known dataset ID OR a raw text to score
        if not headline_id and not text:
            raise ValueError("Provide headline_id or text")

    # Path A: Known dataset item (ID)
        if headline_id:
        # Guard against bad IDs to avoid a KeyError crash
            if headline_id not in self.idx:
                raise KeyError(f"Unknown headline_id: {headline_id}")

            item = self.idx[headline_id]

        # Pull the baked ML confidence (from ml_bake_confidence.py)
            conf_ml = item.get("confidence", 0.5)
            try:
                conf_ml = float(conf_ml)
            except Exception:
                conf_ml = 0.5

        # Decide how to combine ML with heuristic
            if USE_ML_ONLY:
            # Use ONLY the ML confidence for known IDs
                conf = conf_ml
                reasons = []   # We can still build a rationale from item/source cues below
            else:
            # Blend: ML gets the heavier weight; heuristic refines it a bit at runtime
                conf_h, reasons = self._heuristic_conf(
                    item["text"],
                    item.get("source", ""),
                    item.get("domain", "news")
            )
                conf = (ML_BLEND * conf_ml) + ((1.0 - ML_BLEND) * conf_h)

        # Turn the collected reasons (if any) into a human-readable rationale
            rationale = self._mk_rationale(item, reasons)

        # Return the final verdict for a known item
            return {
                "truth": bool(item.get("truth", False)),   # Ground-truth label from dataset
                "confidence": round(float(conf), 3),       # Probability-like score (0–1)
                "rationale": rationale                     # Human explanation
            }


    # Path B: Raw ad-hoc text (no dataset ID)
        conf_h, reasons = self._heuristic_conf(text, "", "news")
        rationale = self._mk_rationale(None, reasons)

    # Threshold the heuristic to infer a provisional truth value
        inferred_truth = conf_h >= 0.6

        return {
            "truth": bool(inferred_truth),
            "confidence": round(float(conf_h), 3),
            "rationale": rationale
        }

_engine = None

def _get_engine():
    # Create one AccuracyEngine and reuse it (avoids reloading file every call)
    global _engine
    if _engine is None:
        _engine = AccuracyEngine()
    return _engine

def get_headline(domain=None):
    # Public function Person B can import directly
    return _get_engine().get_headline(domain)

def check_truth(headline_id=None, text=None):
    # Public function Person B can import directly
    return _get_engine().evaluate(headline_id=headline_id, text=text)

def get_headline_text(headline_id):
    """
    Returns the text of a headline given its ID.
    This is used by the backend (e.g. to send text to OpenAI).
    """
    engine = _get_engine()
    # Try to fetch from in-memory index
    item = engine.idx.get(headline_id)
    if item is not None:
        return {"text": item.get("text", "Headline text missing"),"article preview":item.get("article_preview","")}
    return {"text":"Headline not found", "article_preview":""}
