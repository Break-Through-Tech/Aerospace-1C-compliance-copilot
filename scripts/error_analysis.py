import json
import sys
from pathlib import Path

import pandas as pd

# Paths and column names
GOLD_CSV = "data/benchmark/requirements.csv"
RETRIEVAL_JSON = "outputs/rag/retrieval-comparison.json"
AGENT_JSON = sys.argv[1] if len(sys.argv) > 1 else "outputs/rag/audit-results.json"
OUT_DIR = Path("outputs/error_analysis")

COL_REQ_ID = "requirement_id"
COL_SWE_ID = "swe_id"
COL_LABEL = "verdict"


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if "results" in data and isinstance(data["results"], list):
        return data["results"]
    raise ValueError(f"Could not find the results list. Top-level keys: {list(data.keys())}")


def norm_verdict(v):
    # Maps label wordings onto three classes: gap, partial, meets
    v = (v or "").strip().lower()
    if "partial" in v:
        return "partial"
    if "gap" in v or "non" in v:
        return "gap"
    if "meets" in v or "compliant" in v:
        return "meets"
    return v


def get_audit(item):
    audit = item.get("audit")
    return audit if isinstance(audit, dict) else item


def swe_ids_from_retrieved(retrieved):
    ids = []
    for r in retrieved or []:
        if isinstance(r, dict):
            ids.append(str(r.get("swe_id") or r.get("SWE ID") or ""))
        else:
            ids.append(str(r))
    return [i for i in ids if i]


gold = pd.read_csv(GOLD_CSV).drop_duplicates(subset=COL_REQ_ID).set_index(COL_REQ_ID)

# Rank of the correct SWE clause per requirement (optional file)
ranks = {}
if Path(RETRIEVAL_JSON).exists():
    for r in load_results(RETRIEVAL_JSON):
        ranks[r["requirement_id"]] = r

rows = []
for item in load_results(AGENT_JSON):
    req_id = item.get("requirement_id")
    if req_id not in gold.index:
        continue

    audit = get_audit(item)
    truth = norm_verdict(gold.loc[req_id, COL_LABEL])
    pred = norm_verdict(audit.get("verdict"))
    expected_swe = str(gold.loc[req_id, COL_SWE_ID])
    cited = [str(c.get("swe_id")) for c in (audit.get("citations") or []) if isinstance(c, dict)]
    retrieved = swe_ids_from_retrieved(item.get("retrieved"))
    error_code = item.get("error_code")
    rank_info = ranks.get(req_id, {})
    correct = truth == pred

    # Rough automatic bucket for errors; confirm by reading each case
    if correct:
        suggested = ""
    elif error_code or item.get("status", "ok") != "ok":
        suggested = "pipeline error"
    elif retrieved and expected_swe not in retrieved:
        suggested = "retrieval failure"
    elif cited and expected_swe not in cited:
        suggested = "citation/grounding problem"
    else:
        suggested = "reasoning failure"

    rows.append({
        "req_id": req_id,
        "requirement_text": gold.loc[req_id, "requirement_text"],
        "ground_truth": truth,
        "agent_verdict": pred,
        "correct": correct,
        "error_type": "" if correct else f"gold {truth} -> agent {pred}",
        "expected_swe": expected_swe,
        "gold_gap_type": gold.loc[req_id, "gap_type"],
        "gold_rationale": gold.loc[req_id, "rationale"],
        "cited_swe": ", ".join(cited),
        "retrieved_swe": ", ".join(retrieved),
        "expected_rank_legacy": rank_info.get("legacy_rank"),
        "expected_rank_normalized": rank_info.get("normalized_rank"),
        "error_code": error_code,
        "agent_reasoning": audit.get("reasoning"),
        "suggested_category": suggested,
        "final_category": "",   # fill in manually
        "notes": "",            # fill in manually
    })

df = pd.DataFrame(rows)
errors = df[~df["correct"]]

OUT_DIR.mkdir(parents=True, exist_ok=True)
stem = Path(AGENT_JSON).parent.name + "_" + Path(AGENT_JSON).stem
df.to_csv(OUT_DIR / f"{stem}_all.csv", index=False)
errors.to_csv(OUT_DIR / f"{stem}_errors.csv", index=False)

print(f"Source: {AGENT_JSON}")
print(f"Evaluated: {len(df)}  Correct: {int(df['correct'].sum())}  Errors: {len(errors)}")
print("\nConfusion matrix (rows = gold, columns = agent):")
print(pd.crosstab(df["ground_truth"], df["agent_verdict"]).to_string())
print("\nBy error type:")
print(errors["error_type"].value_counts().to_string())
print("\nBy suggested category:")
print(errors["suggested_category"].value_counts().to_string())

with open(OUT_DIR / f"{stem}_errors.md", "w") as f:
    for _, r in errors.iterrows():
        f.write(f"## {r['req_id']} ({r['error_type']}; {r['suggested_category']})\n\n")
        f.write(f"**Requirement:** {r['requirement_text']}\n\n")
        f.write(f"**Expected SWE:** {r['expected_swe']} (rank: legacy {r['expected_rank_legacy']}, normalized {r['expected_rank_normalized']}) | **Cited:** {r['cited_swe']}\n\n")
        f.write(f"**Agent reasoning:** {r['agent_reasoning']}\n\n")
        f.write(f"**Gold rationale:** {r['gold_rationale']} (gap type: {r['gold_gap_type']})\n\n---\n\n")

print(f"\nSaved to {OUT_DIR}/")