import json, re, random
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer # Converts text into numeric features by counting how often each word appears
from sklearn.linear_model import LogisticRegression # Strong linear model that outputs probabilities for True/False
from sklearn.calibration import CalibratedClassifierCV # Wraps another model (Logistic Regression) and calibrates its probability outputs, such that 0.8 means 80% likely true
from sklearn.model_selection import train_test_split # Splits data into Train and Test so we can measure how well the model performs
from sklearn.metrics import roc_auc_score, accuracy_score # Two ways of checking performance: AUC measures ranking quality, Accuracy measures direct correctness
from scipy.sparse import csr_matrix, hstack # Used for combining sparse matrices efficiently (TF-IDF outputs large, sparse matrices)

random.seed(42)

IN_JSONL = "Data/Headlines.jsonl"
OUT_JSONL = "Data/Headlines.jsonl"

print("[ml_bake_confidence] starting…")

def load_jsonl(path):
    # Reads a JSONL file (1 JSON object per line) into a list of dicts
    rows = []
    line_no = 0

    # Open the file safely
    with open(path, "r", encoding="utf-8") as f:
        # Loop line by line to avoid loading everything at once
        for raw in f:
            line_no += 1
            line = raw.strip()  # ✅ CALL .strip(), don’t assign the method itself

            # Skip empty lines
            if not line:
                continue

            try:
                obj = json.loads(line)  # Convert JSON string → Python dict
            except Exception as e:
                print(f"[load_jsonl] Error at line {line_no}: {e}")
                print(f"[load_jsonl] Offending line (first 150 chars): {line[:150]} ...")
                raise  # Stop so you can see where the malformed line is

            rows.append(obj)
    return rows


def save_jsonl(rows, path):
    # Writes a list of dictionarites back into JSONL
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n") # json.dumps converts Python objects to JSON strings, ensure_ascii=False prevents transforming characters to unicode

def clean_title(s):
    # Basic cleaning for titles: remove excess spaces and cap length
    s = re.sub(r"\s+", " ", str(s or "").strip())
    return s[:240]

def simple_cues(text):
    # Extract features from headlines which may hint reliability or being clickbaited 
    t = str(text or "")
    t_up = t.upper()

    # Count Hedge Words, which weaken certainty & credibility, making claim weaker
    hedge_list = ["may", "might", "could", "reportedly", "allegedly", "sources say"]
    hedge_count = 0
    for word in hedge_list:
        if word in t.lower():
            hedge_count += 1

    # Check for Clickbait Terms, which ruins credibility, given the nature to capture attention is prioritised
    clickbait_terms = ["BREAKING", "SHOCKING", "MUST SEE", "INCREDIBLE"]
    has_clickbait = False
    for term in clickbait_terms:
        if term in t_up:
            has_clickbait = True
            break
    if "!!!" in t:
        has_clickbait = True
    if has_clickbait:
        clicky = 1
    else:
        clicky = 0

    # Headlines with statistics can be considered reliable, due to bringing factual evidence into their discussion
    contains_numbers = re.search(r"\b\d+(\.\d+)?%?\b", t)
    if contains_numbers:
        nums = 1
    else:
        nums = 0

    
    instituition_pattern = re.search(r"\b(SEC|FOMC|ECB|BoE|ONS|IMF)\b", t)
    if instituition_pattern:
        inst = 1
    else:
        inst = 0

    return [hedge_count, clicky, nums, inst]

def main():
    rows = load_jsonl(IN_JSONL)
    if not rows:
        print("No data found.")
        return
    
    print("Loaded", len(rows), "records from", IN_JSONL)

    X_text = [] # Stores cleaned headline texts
    y = [] # Stores labels: 1 for True, 0 For False

    for r in rows:
        # Extract and then clean headline
        headline_text = r.get("text", "")
        headline_text = clean_title(headline_text)

        X_text.append(headline_text)

        if r.get("truth"):
            y.append(1)
        else:
            y.append(0)
    
    print("Prepared", len(X_text), "text samples for training.")

    # Convert text to numeric features using TF-IDF
    print("Building TF-IDF features...")

    vec = TfidfVectorizer(
        ngram_range = (1, 2), # Use 1 or 2 word phrases
        max_features = 5000, # Limit to 5000 most common features
        min_df = 2 # Ignore words that only appear once
    )

    # Fit the vectoriser to our headlines
    X_tfidf = vec.fit_transform(X_text)
    print("TF-IDF matrix shape:", X_tfidf.shape)

    # Create features (hedge words, clickbait ...)

    print("Extracting simple cues...")
    cue_features = []

    for text in X_text:
        cues = simple_cues(text)
        cue_features.append(cues)

    # Convert list of lists into a compressed sparse matrix
    X_cues = csr_matrix(cue_features, dtype="float32")
    print("Cue matrix shape:", X_cues.shape) # Should be (40,000, 4) where the 4 are Hedge, Clickbait, Number, Instituition


    # Combine TF-IDF features with cue features, will have the 5000 features scoreed between 0-1, then have Hedge, Clickbait, Number or Instituition, where they are ranked 0 or 1 (if they have it or not) ; shape becomes (40,000, 5004)

    X = hstack([X_tfidf, X_cues], format="csr") # Compressed Sparse Row format is the fastest and most memory efficiency way to store sparse matrices for ML training ; TF-IDF matrices are 99% zeros (because most headlines don’t contain most words), so CSR makes them efficient to store and process
    print("Combined feature matrix shape:", X.shape)


    # Split data into Training and Test sets

    Xtr, Xte, ytr, yte = train_test_split( # Xtr = training inputs (80%), Xte = test inputs (20%), ytr = training labels, yte = test labels
        X, y, # X = giant matrix of input features (TF-IDF & Cue Matrix), y = list of truth tables: 1 for true, 0 for false, len(X) = len(y) = 40,000
        test_size = 0.2,
        random_state = 42,
        stratify = y # Ensures the proportion of true vs fake headlines are the same in both training and test sets
    )
    print("Training size:", len(ytr), "Test size:", len(yte))


    # Train the Logistic Regression Model
    print("Training model...")
    base = LogisticRegression(
        max_iter = 2000,
        class_weight = "balanced",
        solver = "liblinear",
        n_jobs = 1
    )

    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xtr, ytr)


    # Evaluate Model Performance
    print("Evaluating model...")
    probabilities = clf.predict_proba(Xte)

    y_pred_proba = probabilities[:, 1] # Probability of being True

    auc_score = roc_auc_score(yte, y_pred_proba)
    accuracy = accuracy_score(yte, (y_pred_proba >= 0.5).astype(int))

    print("AUC:", round(auc_score, 3), "| Accuracy:", round(accuracy, 3))


    # Predict confidences for all headlines
    print("Generating final confidences for full dataset...")
    all_probabilities = clf.predict_proba(X)
    all_true_probs = all_probabilities[:, 1]

    for i in range(len(rows)):
        conf_value = float(round(all_true_probs[i], 3))
        rows[i]["confidence"] = conf_value
    

    # Write updated dataset back to file
    save_jsonl(rows, OUT_JSONL)

    print("Wrote updated JSONL with ML confidences →", OUT_JSONL)
    print("Example record:", rows[0])

if __name__ == "__main__":
    main()
