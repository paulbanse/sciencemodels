"""Repair the two dead correlation cells of compare_landscapes_corr-prestige-merit.

Only cells 10 and 11 are touched, and only their source strings: the legacy
camelCase spellings (``Mean_of_Corr_prestige_localMerit`` /
``Mean_of_Corr_prestige_idealMerit``) become the current report schema
(``..._local_merit`` / ``..._ideal_merit``).  The legacy spellings still plot
through the shim in ``sciencemodels.plotting``; the call sites simply no longer
depend on it.

Guard rails, all asserted:

- the notebook round-trips byte-identically through
  ``json.dumps(indent=1, sort_keys=True)`` before the edit, so writing it back
  cannot reformat the rest of the file;
- every other cell (and every non-source key of the edited cells) is compared
  before/after and must be unchanged;
- the edit is idempotent: a second run finds nothing to replace.

Writes: ``notebooks/compare_landscapes_corr-prestige-merit.ipynb`` only.

Run from anywhere:

    /opt/hermes/.venv/bin/python scripts/shimfix/fix_notebook_cells.py
"""
from __future__ import annotations

import difflib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB = os.path.join(REPO, "notebooks", "compare_landscapes_corr-prestige-merit.ipynb")

# cell index -> {legacy token: current token}
FIXES = {
    10: {
        "Mean_of_Corr_prestige_localMerit": "Mean_of_Corr_prestige_local_merit",
    },
    11: {
        "Mean_of_Corr_prestige_idealMerit": "Mean_of_Corr_prestige_ideal_merit",
    },
}


def dump(nb):
    return json.dumps(nb, indent=1, sort_keys=True)


def main():
    with open(NB, encoding="utf-8") as f:
        raw = f.read()
    nb = json.loads(raw)

    assert dump(nb) == raw, "notebook does not round-trip byte-identically; refusing to rewrite it"

    before_cells = [json.loads(json.dumps(cell)) for cell in nb["cells"]]
    changed = []
    for index, fixes in FIXES.items():
        cell = nb["cells"][index]
        assert cell["cell_type"] == "code", f"cell {index} is not a code cell"
        source = "".join(cell["source"])
        for legacy, current in fixes.items():
            if legacy not in source:
                assert current in source, f"cell {index}: neither {legacy} nor {current} present"
                continue
            source = source.replace(legacy, current)
            changed.append((index, legacy, current))
        cell["source"] = source.splitlines(keepends=True)

    if not changed:
        print("no change needed: cells already on the current schema")
        return

    # nothing outside the declared cells may differ
    for index, cell in enumerate(nb["cells"]):
        if index in FIXES:
            assert {k: v for k, v in cell.items() if k != "source"} == {
                k: v for k, v in before_cells[index].items() if k != "source"
            }, f"cell {index}: non-source keys changed"
        else:
            assert cell == before_cells[index], f"cell {index} changed but is not declared"

    new = dump(nb)
    with open(NB, "w", encoding="utf-8") as f:
        f.write(new)

    assert json.dumps(json.load(open(NB, encoding="utf-8")), indent=1, sort_keys=True) == new

    for line in difflib.unified_diff(
        raw.splitlines(), new.splitlines(), "before", "after", n=1, lineterm=""
    ):
        print(line)
    print(f"edited cells: {sorted(FIXES)}; replacements: {len(changed)}")
    for index, legacy, current in changed:
        print(f"  cell {index}: {legacy} -> {current}")


if __name__ == "__main__":
    main()
