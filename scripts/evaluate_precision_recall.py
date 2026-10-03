import pandas as pd
import random

# 1. Load the gold-set CSV
df = pd.read_csv("oct_milestone1/Synthetic Software Requirements BTT 2026 - Sheet1.csv")

print(df.shape)
print(df["Ground Truth Label"].value_counts())

# 2. Placeholder retrieval function
# Swap out once retrieval team's real function is ready
# Takes requirement text, returns a list of top-k retrieved SWE IDs
ALL_SWE_IDS = df["SWE ID"].unique().tolist()

def fake_retrieve(requirement_text, k=5):
    return random.sample(ALL_SWE_IDS, min(k, len(ALL_SWE_IDS)))

# 3. Retrieval precision/recall
# Checks whether the correct SWE ID shows up in the top-k retrieved results
def evaluate_retrieval(data, retrieve_fn, k=5):
    precisions, recalls = [], []
    for _, row in data.iterrows():
        query = row["Synthetic Software Requirement (NASA Project)"]
        relevant = {row["SWE ID"]}

        retrieved = set(retrieve_fn(query, k=k))
        hits = len(retrieved & relevant)

        precisions.append(hits / k)
        recalls.append(hits / len(relevant) if relevant else 1.0)

    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)

# 4. Gap-detection precision/recall
# precision = % of flagged gaps that are real gaps
# recall = % of real gaps that got flagged
# Needs real verdicts from the Pydantic/LLM pipeline, placeholder below just
# randomly guesses, so these numbers aren't meaningful yet
def fake_verdict(requirement_text):
    return random.choice(["compliant", "non_compliant", "insufficient_info"])

def evaluate_gap_detection(data, verdict_fn):
    tp = fp = fn = 0
    for _, row in data.iterrows():
        truth = row["Ground Truth Label"]  # "Compliant" or "Gap"
        pred = verdict_fn(row["Synthetic Software Requirement (NASA Project)"])

        flagged = pred == "non_compliant"
        is_gap = truth == "Gap"

        if flagged and is_gap:
            tp += 1
        elif flagged and not is_gap:
            fp += 1
        elif not flagged and is_gap:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    return precision, recall

if __name__ == "__main__":
    r_precision, r_recall = evaluate_retrieval(df, fake_retrieve, k=5)
    print(f"[Retrieval] Precision@5: {r_precision:.3f}  Recall@5: {r_recall:.3f}")

    g_precision, g_recall = evaluate_gap_detection(df, fake_verdict)
    print(f"[Gap Detection] Precision: {g_precision:.3f}  Recall: {g_recall:.3f}")