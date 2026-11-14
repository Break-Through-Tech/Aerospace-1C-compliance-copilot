
import json
from pathlib import Path

# Find the repository root
ROOT = Path(__file__).resolve().parents[1]

# Paths to existing benchmark results
baseline_path = ROOT / "outputs/rag/baseline.json"
agent_path = ROOT / "outputs/agent/demo.json"

# Load the saved results
with open(baseline_path) as file:
    baseline = json.load(file)

with open(agent_path) as file:
    agent = json.load(file)

# Collect requirement IDs
baseline_ids = {
    item["requirement_id"]
    for item in baseline["results"]
}

agent_ids = {
    item["requirement_id"]
    for item in agent["results"]
}

# Compare the results
print("Baseline requirements:", len(baseline_ids))
print("Agent requirements:", len(agent_ids))

print("Matching requirements:",
      len(baseline_ids & agent_ids))

print("Only in Baseline:",
      sorted(baseline_ids - agent_ids))

print("Only in Agent:",
      sorted(agent_ids - baseline_ids))

# Check whether the requirements match
if baseline_ids == agent_ids:
    print("PASS: Benchmark requirements match.")
else:
    print("FAIL: Benchmark requirements do not match.")
