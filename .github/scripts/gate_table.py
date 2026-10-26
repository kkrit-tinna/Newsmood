"""Render a quality/{date}.json file as a markdown gate table (T4.4).

Used by daily.yml for the job summary. Formats the values T3.5 wrote; it
computes nothing.

Usage: python gate_table.py reports/quality/2026-10-26.json
"""

import json
import sys


def _cell(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        return "`" + json.dumps(value, sort_keys=True) + "`"
    return str(value)


def main(path: str) -> None:
    with open(path) as f:
        quality = json.load(f)
    print("| Gate | Kind | Observed | Threshold | Status |")
    print("|---|---|---|---|---|")
    for g in quality["gates"]:
        status = f"**{g['status']}**" if g["status"] in ("fail", "fired") else g["status"]
        print(f"| `{g['name']}` | {g['kind']} | {_cell(g['observed'])} | {_cell(g['threshold'])} | {status} |")


if __name__ == "__main__":
    main(sys.argv[1])
