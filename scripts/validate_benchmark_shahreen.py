"""Independent benchmark validation for the NASA compliance dataset."""

import csv
import json
from collections import Counter
from pathlib import Path


REQUIRED_COLUMNS = {
    "id",
    "category",
    "requirement_text",
    "swe_id",
    "npr_clause",
    "verdict",
    "rationale",
    "gap_type",
}

ALLOWED_VERDICTS = {
    "Meets",
    "Partial",
    "Gap",
}

def load_csv(path: Path) -> list[dict]:
    """Load a CSV file and return each row as a dictionary."""

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))

def load_json(path: Path):
    """Load and return data from a JSON file."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)

def validate_required_columns(
    rows: list[dict],
) -> list[str]:
    """Check that the benchmark contains all required columns."""

    if not rows:
        return ["Benchmark contains no records."]

    columns = set(rows[0].keys())
    missing = REQUIRED_COLUMNS - columns

    if missing:
        return [
            "Missing required columns: "
            + ", ".join(sorted(missing))
        ]

    return []
