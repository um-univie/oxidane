"""Audit every shard, including failed jobs; do not equate absence of errors with completion."""
import argparse
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("--shards", type=int, default=1000)
parser.add_argument("--expected-records", type=int, default=124598147)
args = parser.parse_args()
# Every counter a shard must report; the parent checks are reported too, as
# zeros, when the sweep ran without `--parent on`.
expected_counters = {
    "unparsed by this crate", "normalized", "no answer (should be 0)",
    "formula changed", "output did not reparse", "not idempotent", "key changed",
    "did not converge", "panics", "skipped as oversize",
    "canonical_formula", "canonical_reparse", "canonical_idempotence",
    "canonical_permutation", "tautomer_permutation", "key_permutation",
    "key_canonical", "record_key_permutation", "record_key_canonical", "record_key",
    "parent_reparse", "parent_idempotence", "parent_permutation", "parent_tautomer",
    "parent_atoms", "impossible valence",
}
# Counted but not findings: what was normalized, and the deposited records
# whose valences no element allows, which are not molecules to normalize.
not_findings = {"normalized", "impossible valence"}
totals = {}
incomplete = []
failures = []
complete = 0
records = 0
for index in range(args.shards):
    path = args.run_dir / "logs" / f"{index}.out"
    output = path.read_text() if path.exists() else ""
    footer = re.search(r"SHARD_COMPLETE shard=\S+ expected=(\d+) exit=(\d+)", output)
    report = re.search(r"^(\d+) lines in .*", output, re.M)
    counters = dict((name.strip(), int(value)) for name, value in
                    re.findall(r"^  ([^\n]+?)\s*:\s*(\d+)\s*$", output, re.M))
    if (not footer or not report or int(footer[1]) != int(report[1])
            or int(footer[2]) not in (0, 1) or set(counters) != expected_counters):
        incomplete.append(index)
        continue
    complete += 1
    records += int(report[1])
    for name, value in counters.items():
        totals[name] = totals.get(name, 0) + value
    if any(value for name, value in counters.items() if name not in not_findings):
        failures.append(index)
result = dict(completed_shards=complete, expected_shards=args.shards,
              records=records, expected_records=args.expected_records,
              incomplete_shards=incomplete, shards_with_findings=failures,
              counters=totals)
result["clean"] = not incomplete and not failures and records == args.expected_records
(args.run_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["clean"] else 1)
