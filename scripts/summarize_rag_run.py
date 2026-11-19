"""Score saved audits after inference, keeping failures in the denominator."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compliance_copilot.schemas import AuditResult


def summarize(report, benchmark, *, require_all=False):
    """Score one report. With require_all, every benchmark ID must be present."""
    cases = {row["requirement_id"]: row for row in benchmark}
    details = []
    seen = set()
    for raw in report["results"]:
        result = AuditResult.model_validate(raw)
        key = result.requirement_id
        if key in seen or key not in cases:
            raise ValueError(f"Duplicate or unknown requirement ID: {key}")
        seen.add(key)
        expected = cases[key]
        if result.requirement_text != expected["requirement_text"]:
            raise ValueError(f"Requirement text differs from benchmark: {key}")
        valid = result.status == "ok"
        predicted = result.audit.verdict if valid else None
        cited = valid and expected["swe_id"] in {c.swe_id for c in result.audit.citations}
        details.append({
            "requirement_id": key, "expected_verdict": expected["verdict"],
            "expected_swe_id": expected["swe_id"], "predicted_verdict": predicted,
            "valid_output": valid, "error_code": result.error_code,
            "expected_clause_retrieved": expected["swe_id"] in {c.swe_id for c in result.retrieved},
            "expected_clause_cited": cited,
            "verdict_matches": valid and predicted == expected["verdict"],
            "grounded_verdict_matches": cited and predicted == expected["verdict"],
        })
    if not details:
        raise ValueError("Report contains no results")
    # A partial run would otherwise shrink the denominator without warning.
    missing = [key for key in cases if key not in seen]
    if require_all and missing:
        raise ValueError(f"Missing {len(missing)} benchmark requirement IDs: {', '.join(missing)}")
    metrics = ["valid_output", "expected_clause_retrieved", "expected_clause_cited",
               "verdict_matches", "grounded_verdict_matches"]
    return {"total": len(details), "counts": {m: sum(r[m] for r in details) for m in metrics},
            "missing_ids": missing, "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-all", action="store_true",
                        help="Fail unless every benchmark requirement ID has a result")
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    if any(report["config"] != reports[0]["config"] for report in reports[1:]):
        parser.error("Combined reports must use the same configuration")
    combined = {"results": [row for report in reports for row in report["results"]]}
    with (ROOT / "data/benchmark/requirements.csv").open(encoding="utf-8", newline="") as stream:
        try:
            summary = summarize(combined, list(csv.DictReader(stream)), require_all=args.require_all)
        except ValueError as exc:
            parser.error(str(exc))
    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
