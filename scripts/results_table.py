"""Print the test-set results table from an executed copy of notebook.ipynb.

Reads the ranking the notebook prints ("1. Random Forest: F1=..., AUC-ROC=...")
so the README table comes straight from the run.
"""
import json
import re
import sys

LINE = re.compile(
    r"^\s*\d+\. (?P<model>[^:]+): F1=(?P<f1>[\d.]+), AUC-ROC=(?P<auc>[\d.]+), "
    r"Prec=(?P<prec>[\d.]+), Rec=(?P<rec>[\d.]+)"
)


def main(path: str) -> None:
    nb = json.load(open(path))
    rows = []
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            for line in "".join(out.get("text", [])).splitlines():
                m = LINE.match(line)
                if m:
                    rows.append(m.groupdict())
    if not rows:
        sys.exit("no ranking found; did the notebook run to the end?")
    print("| Model | F1 | ROC AUC | Precision | Recall |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['f1']} | {r['auc']} | {r['prec']} | {r['rec']} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "notebook.executed.ipynb")
