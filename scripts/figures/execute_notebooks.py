"""Headless execution harness for the figures-v1 notebook validation (ADR-0002).

Every notebook calls ``plt.show()``; under ``MPLBACKEND=Agg`` that produces no
output at all, so the six notebooks would execute without leaving a figure
behind. This harness therefore:

- builds a throwaway mirror (``/tmp/figs_exec``) of the repo: copies of
  ``notebooks/`` plus symlinks to ``sciencemodels/``, ``data/``, ``docs/`` and
  ``scripts/``, so the notebooks' repo-root bootstrap resolves the real tree
  and *no repo file is written by the execution itself*;
- prepends one in-memory capture cell that swaps ``plt.show`` for a savefig
  into ``data/figures_v1/figures/<notebook>_fig<NN>.png`` (the cell is removed
  again before the executed notebook is saved as an artifact);
- executes with ``allow_errors`` so one broken cell does not hide the rest of
  the notebook, and records every failing cell verbatim.

Writes: ``data/figures_v1/figures/*.png``,
``data/figures_v1/executed/<notebook>.ipynb``,
``data/figures_v1/notebook_run_manifest.json``.

Run from anywhere:

    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/figures/execute_notebooks.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time

import nbformat
from nbclient import NotebookClient

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "data", "figures_v1")
FIG_DIR = os.path.join(OUT_DIR, "figures")
EXEC_DIR = os.path.join(OUT_DIR, "executed")
MIRROR = "/tmp/figs_exec"

CAPTURE_CELL = '''
# injected by scripts/figures/execute_notebooks.py -- save show() to PNG under Agg
import os as _os
import hashlib as _hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt

_FIGS_DIR = _os.environ["FIGS_OUT"]
_FIGS_PREFIX = _os.environ["FIGS_PREFIX"]
_FIGS_COUNT = [0]
_FIGS_SKIPPED = []


def _figs_show(*args, **kwargs):
    # provenance: the running cell's source is the last IPython history entry
    try:
        _src = get_ipython().user_ns["In"][-1]
        _tag = _hashlib.md5(_src.encode()).hexdigest()[:8]
    except Exception:
        _tag = "notag"
    for _num in _plt.get_fignums():
        _fig = _plt.figure(_num)
        if not any(_ax.has_data() for _ax in _fig.get_axes()):
            # orphan frame left behind by a cell that failed before plotting
            _FIGS_SKIPPED.append(_tag)
            continue
        _path = _os.path.join(
            _FIGS_DIR, "%s_fig%02d_%s.png" % (_FIGS_PREFIX, _FIGS_COUNT[0], _tag)
        )
        _fig.savefig(_path, dpi=110, bbox_inches="tight")
        _FIGS_COUNT[0] += 1
    _plt.close("all")


_plt.show = _figs_show
print("figure capture armed ->", _FIGS_DIR)
'''


def prepare_mirror():
    if os.path.isdir(MIRROR):
        shutil.rmtree(MIRROR)
    os.makedirs(os.path.join(MIRROR, "notebooks"))
    for entry in ("sciencemodels", "data", "docs", "scripts", "tests"):
        target = os.path.join(REPO, entry)
        if os.path.exists(target):
            os.symlink(target, os.path.join(MIRROR, entry))


def run_one(nb_path, index):
    stem = os.path.splitext(os.path.basename(nb_path))[0]
    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)
    src_cells = list(nb.cells)
    nb.cells = [nbformat.v4.new_code_cell(CAPTURE_CELL)] + src_cells

    os.environ["FIGS_OUT"] = FIG_DIR
    os.environ["FIGS_PREFIX"] = stem
    os.environ["MPLBACKEND"] = "Agg"

    before = set(glob.glob(os.path.join(FIG_DIR, f"{stem}_fig*.png")))
    t0 = time.time()
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": os.path.join(MIRROR, "notebooks")}},
    )
    error = None
    try:
        client.execute()
    except Exception as exc:  # execution-level failure (kernel, timeout)
        error = f"{type(exc).__name__}: {exc}"

    errors = []
    for cell_index, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                ansi = re.compile(r"\x1b\[[0-9;]*m")
                tb = [ansi.sub("", line).strip() for line in (out.get("traceback") or [])]
                tb = [line for line in tb if line]
                errors.append(
                    {
                        "cell": cell_index - 1,  # original index (capture cell removed)
                        "ename": out.get("ename"),
                        "evalue": out.get("evalue"),
                        "traceback_tail": tb[-3:],
                        "last_traceback_line": tb[-1] if tb else None,
                        "cell_source_head": "\n".join(
                            src_cells[cell_index - 1].source.splitlines()[:4]
                        )
                        if cell_index - 1 < len(src_cells)
                        else None,
                    }
                )

    os.makedirs(EXEC_DIR, exist_ok=True)
    nb.cells = nb.cells[1:]  # drop the injected capture cell from the artifact
    executed_path = os.path.join(EXEC_DIR, f"{stem}.ipynb")
    with open(executed_path, "w") as f:
        nbformat.write(nb, f)

    after = set(glob.glob(os.path.join(FIG_DIR, f"{stem}_fig*.png")))
    # filename tag -> cell index, so every figure carries its producing cell
    tag_map = {}
    for idx, cell in enumerate(src_cells):
        if cell.cell_type != "code":
            continue
        for variant in (cell.source, cell.source.strip()):
            tag_map[hashlib.md5(variant.encode()).hexdigest()[:8]] = idx
    figures = []
    for p in sorted(after - before):
        tag = os.path.basename(p).rsplit("_", 1)[-1].split(".")[0]
        figures.append({"path": os.path.relpath(p, REPO), "cell": tag_map.get(tag, "unmapped")})
    return {
        "notebook": os.path.relpath(nb_path, REPO),
        "order": index,
        "executed": error is None and not errors,
        "fatal": error,
        "seconds": round(time.time() - t0, 1),
        "n_code_cells": sum(1 for c in src_cells if c.cell_type == "code"),
        "figures": figures,
        "n_figures": len(figures),
        "errors": errors,
        "executed_nb": os.path.relpath(executed_path, REPO),
    }


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    prepare_mirror()
    notebooks = sorted(glob.glob(os.path.join(REPO, "notebooks", "*.ipynb")))
    results = []
    for i, nb_path in enumerate(notebooks):
        print(f"== [{i + 1}/{len(notebooks)}] {os.path.basename(nb_path)}", flush=True)
        res = run_one(nb_path, i)
        print(
            f"   executed={res['executed']} figures={res['n_figures']} "
            f"errors={len(res['errors'])} ({res['seconds']}s)",
            flush=True,
        )
        results.append(res)
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notebooks_gross": len(notebooks),
        "notebooks_net_clean": sum(r["executed"] for r in results),
        "figures_total": sum(r["n_figures"] for r in results),
        "results": results,
    }
    with open(os.path.join(OUT_DIR, "notebook_run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
