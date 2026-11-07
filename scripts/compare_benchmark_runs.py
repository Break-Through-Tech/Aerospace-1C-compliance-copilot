"""Compare a saved Baseline RAG run with a saved Agent run on the same benchmark.

Inputs are reports written by scripts/run_rag_demo.py and scripts/run_agent_demo.py.
Nothing is re-run. Every benchmark ID must be present in both reports, and failed
outputs stay in every denominator: an invalid output counts as a wrong verdict and as
"not flagged" for gap detection.

Example:
    python scripts/compare_benchmark_runs.py \
        --baseline outputs/rag/audit-results.json \
        --agent outputs/agent/agent-full.json \
        --output-dir outputs/benchmark
"""

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_rag_run import summarize

LABELS = ("Meets", "Partial", "Gap")
# Gap detection treats any non-compliance as the positive class.
POSITIVE = {"Partial", "Gap"}
SYSTEMS = ("baseline", "agent")
AGENT_MODE = "tool_calling_agent"
BENCHMARK_PATH = "data/benchmark/requirements.csv"

# Agent-only settings, and a cache that must not change results.
CONFIG_KEYS_ALLOWED_TO_DIFFER = {"mode", "max_steps", "batch_size", "cache"}
# Hardware and library versions can change greedy output, but are reported rather than refused.
ENVIRONMENT_KEYS = {"device", "python", "packages"}

POLICY = {
    "denominator": "all benchmark IDs in the comparison; failures are never dropped",
    "invalid_output": "counted as a wrong verdict and as not flagged (negative) for gap detection",
    "gap_detection_positive_class": sorted(POSITIVE),
    "undefined_precision_or_recall": "reported as 0.0 (same as scikit-learn zero_division=0)",
    "confidence_intervals": "percentile bootstrap over requirement IDs, paired across systems",
    "paired_test": "exact two-sided McNemar test on per-ID correctness",
    "verdict_agreement": "share of IDs with the same predicted verdict; two invalid outputs count as agreeing",
}


# ---------------------------------------------------------------- loading and checks

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_benchmark(ids=None):
    with (ROOT / BENCHMARK_PATH).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if ids is None:
        return rows
    known = {row["requirement_id"] for row in rows}
    unknown = [i for i in ids if i not in known]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if unknown or duplicated:
        raise ValueError(f"IDs must be unique benchmark IDs; unknown: {unknown}, duplicated: {duplicated}")
    wanted = set(ids)
    return [row for row in rows if row["requirement_id"] in wanted]


def restrict_report(report, benchmark):
    """Keep only rows for the selected benchmark IDs (used with --ids)."""
    wanted = {row["requirement_id"] for row in benchmark}
    restricted = dict(report)
    restricted["results"] = [r for r in report["results"] if r["requirement_id"] in wanted]
    if "runs" in report:
        restricted["runs"] = [r for r in report["runs"] if r["result"]["requirement_id"] in wanted]
    return restricted


def check_configs(baseline_config, agent_config):
    """Refuse runs that differ in anything that defines the system under test."""
    keys = (set(baseline_config) | set(agent_config)) - CONFIG_KEYS_ALLOWED_TO_DIFFER
    mismatched = [k for k in sorted(keys - ENVIRONMENT_KEYS)
                  if baseline_config.get(k) != agent_config.get(k)]
    if mismatched:
        details = "; ".join(f"{k}: baseline={baseline_config.get(k)!r}, agent={agent_config.get(k)!r}"
                            for k in mismatched)
        raise ValueError(f"Runs are not comparable, configuration differs: {details}")
    return [f"Environment differs in '{k}': baseline={baseline_config.get(k)!r}, "
            f"agent={agent_config.get(k)!r}"
            for k in sorted(keys & ENVIRONMENT_KEYS) if baseline_config.get(k) != agent_config.get(k)]


def check_roles(baseline_report, agent_report):
    """Catch swapped inputs, which would silently flip the sign of every difference."""
    if baseline_report["config"].get("mode") == AGENT_MODE or "runs" in baseline_report:
        raise ValueError("The baseline report is a tool-calling Agent run; check --baseline and --agent")
    if agent_report["config"].get("mode") != AGENT_MODE or "runs" not in agent_report:
        return ["The agent report is not a tool-calling Agent run; agent process statistics are unavailable"]
    return []


def check_input_hashes(baseline_hashes, agent_hashes):
    """Shared source and data files should be identical. Line-ending changes also show up here."""
    return [f"Input file hash differs between runs: {name}"
            for name in sorted(set(baseline_hashes) & set(agent_hashes))
            if baseline_hashes[name] != agent_hashes[name]]


# ---------------------------------------------------------------- per-requirement records

def retrieval_queries(run):
    return [step.get("observation", {}).get("query")
            for step in run["steps"] if step.get("tool") == "retrieve_clause"]


def check_runs(report, rows):
    """Agent traces must describe exactly the scored results, one trace per ID."""
    runs = {}
    for run in report.get("runs", []):
        key = run["result"]["requirement_id"]
        if key in runs:
            raise ValueError(f"Duplicate agent run for requirement ID: {key}")
        runs[key] = run
    if "runs" in report:
        if set(runs) != set(rows):
            raise ValueError("Agent 'runs' and 'results' cover different requirement IDs")
        differing = [key for key in rows if runs[key]["result"] != rows[key]]
        if differing:
            raise ValueError(f"Agent run result differs from its 'results' row: {', '.join(differing)}")
    return runs


def score_report(report, benchmark):
    """Return one record per benchmark ID, in benchmark order."""
    summary = summarize(report, benchmark, require_all=True)
    details = {d["requirement_id"]: d for d in summary["details"]}
    rows = {r["requirement_id"]: r for r in report["results"]}
    runs = check_runs(report, rows)
    records = []
    for case in benchmark:
        key = case["requirement_id"]
        record = dict(details[key])
        record["raw_output"] = rows[key].get("raw_output")
        run = runs.get(key)
        if run is not None:
            queries = retrieval_queries(run)
            record["termination"] = run["termination"]
            record["steps"] = len(run["steps"])
            record["tool_sequence"] = " > ".join(str(step.get("tool")) for step in run["steps"])
            # Any search other than the single original-requirement search is a refinement.
            record["refined_query"] = (len(queries) > 1
                                       or any(q not in (None, case["requirement_text"]) for q in queries))
        records.append(record)
    return records


# ---------------------------------------------------------------- metrics

def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def f1(precision, recall):
    return ratio(2 * precision * recall, precision + recall)


def rate(records, field):
    return ratio(sum(bool(r[field]) for r in records), len(records))


def is_flagged(record):
    return record["predicted_verdict"] in POSITIVE


def gap_detection(records):
    tp = sum(r["expected_verdict"] in POSITIVE and is_flagged(r) for r in records)
    fp = sum(r["expected_verdict"] not in POSITIVE and is_flagged(r) for r in records)
    fn = sum(r["expected_verdict"] in POSITIVE and not is_flagged(r) for r in records)
    tn = len(records) - tp - fp - fn
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1(precision, recall)}


def per_class(records):
    result = {}
    for label in LABELS:
        tp = sum(r["expected_verdict"] == label and r["predicted_verdict"] == label for r in records)
        fp = sum(r["expected_verdict"] != label and r["predicted_verdict"] == label for r in records)
        # An invalid prediction is a miss for its gold class but a false positive for no class.
        fn = sum(r["expected_verdict"] == label and r["predicted_verdict"] != label for r in records)
        precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
        result[label] = {"support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                         "precision": precision, "recall": recall, "f1": f1(precision, recall)}
    result["macro"] = {m: sum(result[label][m] for label in LABELS) / len(LABELS)
                       for m in ("precision", "recall", "f1")}
    return result


def confusion(records):
    columns = LABELS + ("Invalid",)
    counts = Counter((r["expected_verdict"], r["predicted_verdict"] or "Invalid") for r in records)
    return {gold: {pred: counts[(gold, pred)] for pred in columns} for gold in LABELS}


# Headline metrics, each a function of a list of records (used for point estimates and bootstrap).
HEADLINE = {
    "valid_output_rate": lambda rs: rate(rs, "valid_output"),
    "clause_retrieval_accuracy": lambda rs: rate(rs, "expected_clause_retrieved"),
    "clause_citation_accuracy": lambda rs: rate(rs, "expected_clause_cited"),
    "verdict_accuracy": lambda rs: rate(rs, "verdict_matches"),
    "grounded_verdict_accuracy": lambda rs: rate(rs, "grounded_verdict_matches"),
    "gap_precision": lambda rs: gap_detection(rs)["precision"],
    "gap_recall": lambda rs: gap_detection(rs)["recall"],
    "gap_f1": lambda rs: gap_detection(rs)["f1"],
    "macro_f1": lambda rs: per_class(rs)["macro"]["f1"],
}
COUNTED = {
    "valid_output_rate": "valid_output",
    "clause_retrieval_accuracy": "expected_clause_retrieved",
    "clause_citation_accuracy": "expected_clause_cited",
    "verdict_accuracy": "verdict_matches",
    "grounded_verdict_accuracy": "grounded_verdict_matches",
}


def percentile(sorted_values, q):
    position = q * (len(sorted_values) - 1)
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def bootstrap(baseline, agent, *, resamples, seed):
    """Paired percentile bootstrap: both systems are resampled on the same IDs."""
    rng = random.Random(seed)
    size = len(baseline)
    draws = {name: {"baseline": [], "agent": [], "difference": []} for name in HEADLINE}
    for _ in range(resamples):
        index = [rng.randrange(size) for _ in range(size)]
        sample_b = [baseline[i] for i in index]
        sample_a = [agent[i] for i in index]
        for name, metric in HEADLINE.items():
            value_b, value_a = metric(sample_b), metric(sample_a)
            draws[name]["baseline"].append(value_b)
            draws[name]["agent"].append(value_a)
            draws[name]["difference"].append(value_a - value_b)
    intervals = {}
    for name, series in draws.items():
        intervals[name] = {}
        for key, values in series.items():
            values.sort()
            intervals[name][key] = [percentile(values, 0.025), percentile(values, 0.975)]
    return intervals


def mcnemar_exact(baseline_only, agent_only):
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    discordant = baseline_only + agent_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(baseline_only, agent_only) + 1))
    return min(1.0, 2 * tail / 2 ** discordant)


def paired(baseline, agent):
    result = {
        "verdict_agreement": ratio(sum(b["predicted_verdict"] == a["predicted_verdict"]
                                       for b, a in zip(baseline, agent)), len(baseline)),
        "identical_raw_output": sum(b["raw_output"] is not None and b["raw_output"] == a["raw_output"]
                                    for b, a in zip(baseline, agent)),
    }
    for field in ("verdict_matches", "grounded_verdict_matches"):
        both = sum(b[field] and a[field] for b, a in zip(baseline, agent))
        baseline_only = sum(b[field] and not a[field] for b, a in zip(baseline, agent))
        agent_only = sum(a[field] and not b[field] for b, a in zip(baseline, agent))
        result[f"mcnemar_{field}"] = {
            "both_correct": both, "baseline_only_correct": baseline_only,
            "agent_only_correct": agent_only,
            "both_wrong": len(baseline) - both - baseline_only - agent_only,
            "p_value": mcnemar_exact(baseline_only, agent_only),
        }
    return result


def process_stats(records):
    """Agent orchestration evidence. Empty for reports without 'runs'."""
    runs = [r for r in records if "termination" in r]
    if not runs:
        return None
    steps = [r["steps"] for r in runs]
    return {
        "runs": len(runs),
        "terminations": dict(Counter(r["termination"] for r in runs)),
        "steps_mean": sum(steps) / len(steps), "steps_min": min(steps), "steps_max": max(steps),
        "query_refinements": sum(r["refined_query"] for r in runs),
        "tool_sequences": dict(Counter(r["tool_sequence"] for r in runs).most_common()),
    }


def system_metrics(records):
    gold = Counter(r["expected_verdict"] for r in records)
    majority_label, majority_count = gold.most_common(1)[0]
    return {
        "total": len(records),
        "counts": {name: sum(bool(r[field]) for r in records) for name, field in COUNTED.items()},
        "headline": {name: metric(records) for name, metric in HEADLINE.items()},
        "invalid_outputs": sum(not r["valid_output"] for r in records),
        "error_codes": dict(Counter(r["error_code"] for r in records if r["error_code"])),
        "predicted_distribution": dict(Counter(r["predicted_verdict"] or "Invalid" for r in records)),
        "gap_detection": gap_detection(records),
        "per_class": per_class(records),
        "confusion_matrix": confusion(records),
        "majority_class_reference": {"label": majority_label,
                                     "verdict_accuracy": ratio(majority_count, len(records))},
        "agent_process": process_stats(records),
    }


# ---------------------------------------------------------------- comparison and output

def compare(baseline_report, agent_report, benchmark, *, resamples=10000, seed=0):
    warnings = check_roles(baseline_report, agent_report)
    warnings += check_configs(baseline_report["config"], agent_report["config"])
    warnings += check_input_hashes(baseline_report.get("input_sha256", {}),
                                   agent_report.get("input_sha256", {}))
    baseline = score_report(baseline_report, benchmark)
    agent = score_report(agent_report, benchmark)
    records = {"baseline": baseline, "agent": agent}
    return {
        "requirements": len(benchmark),
        "policy": POLICY,
        "warnings": warnings,
        "systems": {name: system_metrics(rs) for name, rs in records.items()},
        "paired": paired(baseline, agent),
        "bootstrap_95ci": bootstrap(baseline, agent, resamples=resamples, seed=seed)
        if resamples else None,
        "bootstrap": {"resamples": resamples, "seed": seed},
        "gold_distribution": dict(Counter(r["verdict"] for r in benchmark)),
    }, records


def per_requirement_rows(benchmark, records):
    fields = ("predicted_verdict", "valid_output", "error_code", "expected_clause_retrieved",
              "expected_clause_cited", "verdict_matches", "grounded_verdict_matches")
    rows = []
    for i, case in enumerate(benchmark):
        b, a = records["baseline"][i], records["agent"][i]
        row = {"requirement_id": case["requirement_id"], "expected_verdict": case["verdict"],
               "expected_swe_id": case["swe_id"]}
        for name, record in (("baseline", b), ("agent", a)):
            for field in fields:
                row[f"{name}_{field}"] = record[field]
        for field in ("termination", "steps", "refined_query", "tool_sequence"):
            row[f"agent_{field}"] = a.get(field)
        row["verdicts_differ"] = b["predicted_verdict"] != a["predicted_verdict"]
        row["correctness_differs"] = b["verdict_matches"] != a["verdict_matches"]
        row["raw_output_identical"] = b["raw_output"] is not None and b["raw_output"] == a["raw_output"]
        rows.append(row)
    return rows


def percent(value):
    return f"{100 * value:.1f}%"


def render_markdown(result, inputs):
    systems = result["systems"]
    ci = result["bootstrap_95ci"]
    total = result["requirements"]
    lines = ["# Baseline vs Agent benchmark comparison", "",
             f"Requirements compared: {total}. Gold verdicts: "
             + ", ".join(f"{k} {v}" for k, v in result["gold_distribution"].items()) + ".", "",
             f"- Baseline: `{inputs['baseline']['path']}`",
             f"- Agent: `{inputs['agent']['path']}`",
             f"- Model: `{inputs['baseline']['config'].get('model')}`, "
             f"top_k={inputs['baseline']['config'].get('top_k')}", ""]

    def cell(name, system):
        value = systems[system]["headline"][name]
        count = systems[system]["counts"].get(name)
        text = f"{count}/{total} ({percent(value)})" if count is not None else f"{value:.3f}"
        if ci:
            low, high = ci[name][system]
            text += f" [{low:.2f}, {high:.2f}]" if count is None else f" [{percent(low)}, {percent(high)}]"
        return text

    lines += ["## Headline metrics (95% bootstrap CI)", "",
              "| Metric | Baseline | Agent | Agent - Baseline |", "| --- | --- | --- | --- |"]
    for name in HEADLINE:
        difference = systems["agent"]["headline"][name] - systems["baseline"]["headline"][name]
        diff_text = f"{difference:+.3f}"
        if ci:
            low, high = ci[name]["difference"]
            diff_text += f" [{low:+.3f}, {high:+.3f}]"
        lines.append(f"| {name.replace('_', ' ')} | {cell(name, 'baseline')} | {cell(name, 'agent')} | {diff_text} |")
    majority = systems["baseline"]["majority_class_reference"]
    lines += ["", f"Reference: always predicting {majority['label']} gives verdict accuracy "
              f"{percent(majority['verdict_accuracy'])}.", ""]

    paired_result = result["paired"]
    lines += ["## Paired comparison", "",
              f"- Verdict agreement: {percent(paired_result['verdict_agreement'])}",
              f"- Identical raw audit output: {paired_result['identical_raw_output']}/{total}"]
    for field in ("verdict_matches", "grounded_verdict_matches"):
        test = paired_result[f"mcnemar_{field}"]
        lines.append(f"- McNemar ({field.replace('_', ' ')}): baseline-only correct "
                     f"{test['baseline_only_correct']}, agent-only correct {test['agent_only_correct']}, "
                     f"p = {test['p_value']:.3f}")
    lines.append("")

    for name in SYSTEMS:
        metrics = systems[name]
        lines += [f"## {name.capitalize()} details", "",
                  "| Gold \\ Predicted | " + " | ".join(LABELS + ("Invalid",)) + " |",
                  "| --- |" + " --- |" * (len(LABELS) + 1)]
        for gold, row in metrics["confusion_matrix"].items():
            lines.append(f"| {gold} | " + " | ".join(str(v) for v in row.values()) + " |")
        lines += ["", "| Class | Support | Precision | Recall | F1 |", "| --- | --- | --- | --- | --- |"]
        for label in LABELS:
            c = metrics["per_class"][label]
            lines.append(f"| {label} | {c['support']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} |")
        gap = metrics["gap_detection"]
        lines += ["", f"Gap detection (Partial or Gap = positive): TP {gap['tp']}, FP {gap['fp']}, "
                  f"FN {gap['fn']}, TN {gap['tn']}.", ""]
        process = metrics["agent_process"]
        if process:
            lines += [f"Agent runs: terminations {process['terminations']}, "
                      f"steps mean {process['steps_mean']:.2f} (min {process['steps_min']}, "
                      f"max {process['steps_max']}), query refinements {process['query_refinements']}.", ""]
    if result["warnings"]:
        lines += ["## Warnings", ""] + [f"- {w}" for w in result["warnings"]] + [""]
    lines += ["Invalid outputs count as wrong verdicts and as not flagged for gap detection. "
              "Benchmark examples are development cases, not an independent holdout.", ""]
    return "\n".join(lines)


def display_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True, help="Report from run_rag_demo.py")
    parser.add_argument("--agent", type=Path, required=True, help="Report from run_agent_demo.py")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/benchmark")
    parser.add_argument("--ids", nargs="+", help="Compare only these benchmark IDs (default: all)")
    parser.add_argument("--bootstrap", type=int, default=10000,
                        help="Bootstrap resamples for confidence intervals; 0 disables them")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("bootstrap must be zero or a positive integer")
    try:
        benchmark = load_benchmark(args.ids)
        reports = {"baseline": load_json(args.baseline), "agent": load_json(args.agent)}
        if args.ids:
            reports = {name: restrict_report(report, benchmark) for name, report in reports.items()}
        result, records = compare(reports["baseline"], reports["agent"], benchmark,
                                  resamples=args.bootstrap, seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    inputs = {name: {"path": display_path(path), "config": reports[name]["config"],
                     "input_sha256": reports[name].get("input_sha256", {})}
              for name, path in (("baseline", args.baseline), ("agent", args.agent))}
    benchmark_sha256 = hashlib.sha256((ROOT / BENCHMARK_PATH).read_bytes()).hexdigest()
    result = {"inputs": inputs, "scoring_sha256": {BENCHMARK_PATH: benchmark_sha256},
              "subset_ids": args.ids, **result}

    # Fixed LF line endings so the same inputs give byte-identical files on every OS.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n",
                                                     encoding="utf-8", newline="\n")
    (args.output_dir / "comparison.md").write_text(render_markdown(result, inputs),
                                                   encoding="utf-8", newline="\n")
    rows = per_requirement_rows(benchmark, records)
    with (args.output_dir / "per_requirement.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({name: {k: round(v, 4) for k, v in result["systems"][name]["headline"].items()}
                      for name in SYSTEMS}, indent=2))
    print("Paired:", json.dumps(result["paired"]))
    for warning in result["warnings"]:
        print("WARNING:", warning)
    print(f"Saved comparison.json, comparison.md, per_requirement.csv to {display_path(args.output_dir)}")


if __name__ == "__main__":
    main()
