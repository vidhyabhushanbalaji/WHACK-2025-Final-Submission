import pandas as pd, json, random, re
from urllib.parse import urlparse

WRITE_FULL_ARTICLE = False  
# Set True if you want to keep full text in JSONL for debug/offline use


random.seed(42)
# Fixes randomness so sampling is reproducible


ATN_CSV   = "Data/all-the-news-2-1.csv"   # All-the-News Dataset (true news)
FAKE_CSV  = "Data/Fake.csv"           # Fake News Dataset (false news)
OUT_JSONL = "Data/Headlines.jsonl"    # Output File

# Faster-Loading
ATN_USECOLS = ["title", "url", "article"]    # only load the columns we actually need
ATN_NROWS   = 20000                          # cap how many ATN rows we read
FAKE_USECOLS = ["title", "text"]             # only needed cols from Fake.csv

def clean_text(s):
    # Normalises whitespace and strips quotes/punctuation
    s = re.sub(r"\s+", " ", str(s or "").strip())
    s = s.replace("“","").replace("”","").replace("’","'")
    return s[:240]

def clean_article(s):
    # Light clean for article/body; allow longer text but cap so JSONL stays small
    s = re.sub(r"\s+", " ", str(s or "").strip())
    s = re.sub(r"<[^>]+>", "", s)  # basic HTML strip if present
    return s[:5000]

def make_preview(s, max_sents=3, hard_cap=400):
    # Keep first 3 sentences, then cap length
    text = str(s or "").strip()
    # Sentence split on ., !, ?
    parts = re.split(r'(?<=[.!?])\s+', text)
    preview = " ".join(parts[:max_sents]).strip()
    if len(preview) > hard_cap:
        preview = preview[:hard_cap].rstrip() + "…"
    return preview


def root_domain(url):
    # Extracts domain name from a full URL
    try:
        host = urlparse(str(url)).netloc.lower() # Converts the url into its network location part (eg "www.vox.com")
    except Exception:
        host = ""
    if not host:
        return "unknown" # If there's still no valid host after parsing, return "unknown" as a placeholder for source
    parts = host.split(".") # Splits the hostname into sections divided by dots (eg ["www","vox","com"])
    if len(parts) >= 3 and parts[-2] in {"co","com","org","gov","ac"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:]) # Handles cases like 'bbc.co.uk' or 'ox.ac.uk', checks if we know 2nd Last & Last Part, & If so we return last three parts joined (eg bbc.co.uk)
    if len(parts) >= 2:
        return ".".join(parts[-2:]) # For normal domains, we can just take the last two parts
    return host # Return the raw host string if domain parsing failed

def guess_domain(title):
    # Categorises news type
    t = (title or "").lower()
    if re.search(r"\b(cpi|gdp|inflation|stock|market|bank|bond|ecb|boe|fed)\b", t):
        return "econ"
    if re.search(r"\b(ai|artificial intelligence|gpt|openai|xai|model|llm)\b", t):
        return "ai"
    if re.search(r"\b(twitter|tiktok|youtube|instagram|meme|viral|influencer)\b", t):
        return "social"
    return "news"

def main():
    # Load datasets & Read only needed columns, and only the first ATN_NROWS rows to keep it fast
    df_t = pd.read_csv(ATN_CSV, usecols=ATN_USECOLS, nrows=ATN_NROWS, low_memory=False)
    df_f = pd.read_csv(FAKE_CSV, usecols=FAKE_USECOLS, low_memory=False)

    # Drop empty/malformed titles early so we don’t process junk
    df_t = df_t.dropna(subset=["title"])
    df_f = df_f.dropna(subset=["title"])

    # Balance the datasets (ATN has 2.5 million values, whilst Fake has 18k values)
    n_fake = len(df_f)
    n_true = min(n_fake, len(df_t))
    df_t = df_t.sample(n_true, random_state=42)
    df_f = df_f.sample(n_fake, random_state=42)  # Explicit sample to randomise Fake order too
    print(f"Balanced dataset: {len(df_t)} true vs {len(df_f)} fake")

    # Build True Records
    true_records = []
    for i, row in enumerate(df_t.itertuples(index=False), 1):
        title = clean_text(getattr(row, "title", ""))
        url   = getattr(row, "url", "")
        source = root_domain(url)
        dom = guess_domain(title)
        article_full = clean_article(getattr(row, "article", ""))
        article_preview = make_preview(article_full)

        rec = {
        "id": f"t{i:05d}",
        "text": title,
        "domain": dom,
        "truth": True,
        "rationale": "Published by verified outlet.",
        "confidence": 0.8,
        "source": source,
        "url": url,
        "article_preview": article_preview
    }

        if WRITE_FULL_ARTICLE:
            rec["article"] = article_full

        true_records.append(rec)
    


    # Build Fake records
    fake_records = []
    for i, row in enumerate(df_f.itertuples(index=False), 1):
        title = clean_text(getattr(row, "title", ""))
        dom = guess_domain(title)
        article_full = clean_article(getattr(row, "text", ""))
        article_preview = make_preview(article_full)

        rec = {
        "id": f"f{i:05d}",
        "text": title,
        "domain": dom,
        "truth": False,
        "rationale": "Flagged as false or misleading claim.",
        "confidence": 0.2,
        "source": "randomblog",
        "url": "",
        "article_preview": article_preview
    }

        if WRITE_FULL_ARTICLE:
            rec["article"] = article_full

        fake_records.append(rec)



    # Merge and write JSONL
    records = true_records + fake_records
    random.shuffle(records)
    # Combines both lists into one dataset, then randomises it so game doesn't show all true or all false

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            # Loops through every record (each headline represented as a Python dictionary)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            # Converts the dictionary into a JSON string and writes it as one line

    print(f"Wrote {len(records)} records → {OUT_JSONL}") # Prints a message confirming how many total headlines were written and where the file is located
    print("Example:", records[0]) # Prints the first record as a sample preview

if __name__ == "__main__":
    main()
    # Prevents it from running automatically if imported by another file
