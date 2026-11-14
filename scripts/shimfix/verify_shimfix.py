"""Verification driver for the correlation-family legacy-column shim (2026-11-13).

Stages (each writes its own checkpoint under ``data/shimfix_v1/`` so a timeout
never loses earlier work):

``units``
    Shim equivalence, computed rather than remembered:
    - the harvest path of the live ``sciencemodels.plotting`` is compared
      frame-for-frame against ``scripts/shimfix/baseline_plotting_snapshot.py``
      (a verbatim copy of the pre-edit module, sha256 pinned below), so the
      harvest-column behaviour is proven byte-for-byte rather than asserted;
    - the correlation alias matrix: six metrics x six spellings x four real
      frames (fresh report CSV, fresh per-step frame, legacy report CSV, legacy
      frame), every resolution must name a column that exists *and* hold the
      same values as the canonical reference column;
    - legacy renames on read keep the aggregation prefix, leave foreign columns
      alone (``Mean_of_Step``, ``mean_prestige``, ``corr_prestige_local_merit_final``)
      and are idempotent.

``cells``
    Executes the three previously failing cells in isolation (bootstrap + the
    cell itself), each with a figure-capture cell, and records per-cell stdout,
    errors and produced figures.  Adds legacy probes that call the plotting
    helpers with the legacy spellings against legacy CSVs, to prove the shim
    covers the old call sites too.

``notebooks``
    Executes all six notebooks end-to-end, exactly like
    ``scripts/figures/execute_notebooks.py`` but writing into
    ``data/shimfix_v1/`` and checkpointing per notebook.

``report``
    Aggregates figure inventory (exists, bytes > 10 kB, imread shape) and the
    six-notebook regression table against the pre-fix manifest.

Writes: ``data/shimfix_v1/**`` only; the throwaway mirror is ``/tmp/shimfix_exec``.
Run from anywhere:

    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/shimfix/verify_shimfix.py units
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/shimfix/verify_shimfix.py cells
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/shimfix/verify_shimfix.py notebooks
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/shimfix/verify_shimfix.py report
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "data", "shimfix_v1")
FIG = os.path.join(OUT, "figures")
EXEC = os.path.join(OUT, "executed")
CKPT = os.path.join(OUT, "checkpoints")
MIRROR = "/tmp/shimfix_exec"

sys.path.insert(0, REPO)
from sciencemodels import plotting as live  # noqa: E402

BASELINE_SHA256 = "5ef483090225870b724ba3213580ea464a2aedf5347958a143b57d1933d52e1e"

FRAMES = {
    "fresh_report": "data/figures_v1/run_number0.csv",
    "fresh_frame": "data/figures_v1/longitudinal_random/run_curiosity0.0_agent_seed0.csv",
    # pre-restructure fixtures, quarantined under data/legacy_2025_2026/ on
    # 2026-11-14 (cleanup round); the shim's whole point is that these still read
    "legacy_report": "data/legacy_2025_2026/testrun_corr-prestige-merit_seed4.csv",
    "legacy_frame": "data/legacy_2025_2026/model_long_random.csv",
}

SIX_NOTEBOOKS = [
    "analyze_data.ipynb",
    "compare_landscapes_Prestige.ipynb",
    "compare_landscapes_corr-prestige-merit.ipynb",
    "compare_landscapes_flexible_search.ipynb",
    "compare_landscapes_normalizedknowledge.ipynb",
    "longitudinalcomparisions.ipynb",
]

CAPTURE_CELL = '''
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
    try:
        _src = get_ipython().user_ns["In"][-1]
        _tag = _hashlib.md5(_src.encode()).hexdigest()[:8]
    except Exception:
        _tag = "notag"
    for _num in _plt.get_fignums():
        _fig = _plt.figure(_num)
        if not any(_ax.has_data() for _ax in _fig.get_axes()):
            _FIGS_SKIPPED.append(_tag)
            continue
        _path = _os.path.join(
            _FIGS_DIR, "%s_fig%02d_%s.png" % (_FIGS_PREFIX, _FIGS_COUNT[0], _tag)
        )
        _fig.savefig(_path, dpi=110, bbox_inches="tight")
        _FIGS_COUNT[0] += 1
    _plt.close("all")


_plt.show = _figs_show
'''


# --------------------------------------------------------------------------- io


def ensure_dirs():
    for d in (OUT, FIG, EXEC, CKPT):
        os.makedirs(d, exist_ok=True)


def checkpoint(name, payload):
    ensure_dirs()
    path = os.path.join(CKPT, name + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def png_info(path):
    size = os.path.getsize(path)
    shape = None
    error = None
    try:
        shape = list(mpimg.imread(path).shape)
    except Exception as exc:  # pragma: no cover - corrupt PNG
        error = f"{type(exc).__name__}: {exc}"
    return {
        "path": os.path.relpath(path, REPO),
        "bytes": size,
        "shape": shape,
        "nontrivial": size > 10240 and error is None,
        "error": error,
    }


# ------------------------------------------------------------------- stage units


def _load_baseline():
    import importlib.util

    path = os.path.join(REPO, "scripts", "shimfix", "baseline_plotting_snapshot.py")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert digest == BASELINE_SHA256, f"baseline snapshot tampered with: {digest}"
    spec = importlib.util.spec_from_file_location("baseline_plotting", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage_units():
    base = _load_baseline()
    out = {"baseline_sha256": BASELINE_SHA256, "frames": {}, "alias_matrix": [], "read_shim": [], "errors": []}

    corr_legacy = {"corr_prestige_localMerit": "Corr_prestige_local_merit",
                   "mean_corr_prestige_localMerit": "Mean_of_Corr_prestige_local_merit"}

    for label, path in FRAMES.items():
        raw = pd.read_csv(os.path.join(REPO, path))
        old = base.read_csv(os.path.join(REPO, path))
        new = live.read_csv(os.path.join(REPO, path))

        # harvest path: live must equal baseline exactly once the corr renames
        # (the only sanctioned delta) are applied to the baseline frame
        old_plus_corr = old.rename(columns={k: v for k, v in corr_legacy.items()
                                            if k in old.columns and v not in old.columns})
        pd.testing.assert_frame_equal(old_plus_corr, new, check_like=False)

        # harvest renames themselves: names, values, dtypes
        harvest = {}
        for old_col, new_col in base.LEGACY_COLUMN_RENAMES.items():
            if old_col in raw.columns:
                assert new_col in new.columns and old_col not in new.columns
                pd.testing.assert_series_equal(raw[old_col], new[new_col], check_names=False)
                assert str(raw[old_col].dtype) == str(new[new_col].dtype)
                harvest[old_col] = new_col

        out["frames"][label] = {
            "path": path,
            "rows": int(len(new)),
            "columns": int(len(new.columns)),
            "harvest_renames": harvest,
            "corr_columns_live": [c for c in new.columns if "corr" in c.lower()],
            "corr_columns_baseline": [c for c in old.columns if "corr" in c.lower()],
            "identical_to_baseline_plus_sanctioned_corr_renames": True,
        }

    # alias matrix over the six metrics, six spellings, four real frames.  A
    # request must resolve to a column that exists, values must be identical
    # inside an aggregation class (per-step frame vs aggregated report row), and
    # the aggregation class of the request must be honoured whenever the frame
    # carries a column of that class.
    for label, path in FRAMES.items():
        df = live.read_csv(os.path.join(REPO, path))
        for key, aliases in live.CORR_METRIC_ALIASES.items():
            family = [c for c in df.columns if live._canonical_corr_name(c) == key]
            if not family:
                # the metric simply is not in this frame (legacy CSVs carry the
                # local variant only); every spelling must keep failing loudly
                for alias in aliases:
                    assert live._resolve_column(df, alias) not in df.columns
                    out["alias_matrix"].append(
                        {"frame": label, "metric": key, "requested": alias,
                         "resolved": None, "metric_absent_in_frame": True}
                    )
                continue
            available = {live._is_aggregated(c) for c in family}
            reference = {}
            for alias in aliases:
                resolved = live._resolve_column(df, alias)
                agg = live._is_aggregated(alias)
                honoured = agg in available
                if not honoured:
                    # the frame carries this metric only in the other aggregation
                    # class: the request must stay unresolved so the caller raises
                    # KeyError instead of plotting a run-mean as a time series
                    assert resolved not in df.columns, (
                        f"{label}: {alias} (class {'mean' if agg else 'step'}) crossed into "
                        f"{resolved} although the frame carries the other class only"
                    )
                    out["alias_matrix"].append(
                        {"frame": label, "metric": key, "requested": alias,
                         "resolved": None, "cross_class_refused": True}
                    )
                    continue
                assert resolved in df.columns, f"{label}: {alias} resolved to missing {resolved}"
                assert live._is_aggregated(resolved) == agg, (
                    f"{label}: {alias} resolved to {resolved} of the wrong aggregation class"
                )
                if agg in reference:
                    pd.testing.assert_series_equal(
                        df[resolved], reference[agg], check_names=False,
                        obj=f"{label}/{alias}",
                    )
                reference[agg] = df[resolved]
                out["alias_matrix"].append(
                    {
                        "frame": label,
                        "metric": key,
                        "requested": alias,
                        "resolved": resolved,
                        "aggregated_request": agg,
                        "aggregated_resolved": live._is_aggregated(resolved),
                        "aggregation_class_honoured": honoured,
                        "values_match_reference": True,
                    }
                )

    # read-direction renames: legacy spellings -> current names only; foreign
    # columns untouched; idempotent
    synthetic = pd.DataFrame(
        {
            "Step": [0, 1],
            "Mean_of_Step": [0.0, 1.0],
            "mean_prestige": [0.1, 0.2],
            "corr_prestige_localMerit": [0.3, 0.4],
            "mean_corr_prestige_localMerit": [0.35, 0.45],
            "Mean_of_Corr_prestige_vanishing_idealMerit": [0.5, 0.6],
            "corr_prestige_local_merit_final": [-0.1, -0.2],
        }
    )
    shimmed = live._apply_legacy_shim(synthetic)
    expected = [
        "Step",
        "Mean_of_Step",
        "mean_prestige",
        "Corr_prestige_local_merit",
        "Mean_of_Corr_prestige_local_merit",
        "Mean_of_Corr_prestige_vanishing_ideal_merit",
        "corr_prestige_local_merit_final",
    ]
    assert list(shimmed.columns) == expected, list(shimmed.columns)
    pd.testing.assert_series_equal(
        shimmed["Corr_prestige_local_merit"], synthetic["corr_prestige_localMerit"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        shimmed["Mean_of_Corr_prestige_local_merit"], synthetic["mean_corr_prestige_localMerit"],
        check_names=False,
    )
    pd.testing.assert_frame_equal(live._apply_legacy_shim(shimmed), shimmed)
    out["read_shim"] = {
        "input_columns": list(synthetic.columns),
        "output_columns": list(shimmed.columns),
        "foreign_columns_untouched": ["Mean_of_Step", "mean_prestige", "corr_prestige_local_merit_final"],
        "idempotent": True,
    }

    # unknown names still raise rather than silently picking a column
    try:
        live.plot_smt("Mean_of_Corr_prestige_not_a_metric", {"x": live.read_csv(os.path.join(REPO, FRAMES["fresh_report"]))})
    except KeyError as exc:
        out["unknown_name_raises_KeyError"] = str(exc)
    else:
        out["errors"].append("unknown corr name did not raise KeyError")

    checkpoint("units", out)
    print(json.dumps({k: v for k, v in out.items() if k != "alias_matrix"}, indent=2))
    print(f"alias resolutions checked: {len(out['alias_matrix'])}")
    return out


# ------------------------------------------------------------------- stage cells


def prepare_mirror():
    if os.path.isdir(MIRROR):
        shutil.rmtree(MIRROR)
    os.makedirs(os.path.join(MIRROR, "notebooks"))
    for entry in ("sciencemodels", "data", "docs", "scripts", "tests"):
        target = os.path.join(REPO, entry)
        if os.path.exists(target):
            os.symlink(target, os.path.join(MIRROR, entry))


def _execute(nb, fig_dir, prefix, timeout=1800):
    import nbformat
    from nbclient import NotebookClient

    os.makedirs(fig_dir, exist_ok=True)
    os.environ["FIGS_OUT"] = fig_dir
    os.environ["FIGS_PREFIX"] = prefix
    os.environ["MPLBACKEND"] = "Agg"

    before = set(glob.glob(os.path.join(fig_dir, f"{prefix}_fig*.png")))
    t0 = time.time()
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": os.path.join(MIRROR, "notebooks")}},
    )
    fatal = None
    try:
        client.execute()
    except Exception as exc:
        fatal = f"{type(exc).__name__}: {exc}"
    # figures are identified by mtime, not by set difference: a rerun rewrites
    # the same deterministic filenames and must still report them
    after = sorted(
        p
        for p in glob.glob(os.path.join(fig_dir, f"{prefix}_fig*.png"))
        if os.path.getmtime(p) >= t0 - 1
    )
    stale = sorted(before - set(after))
    return fatal, after, stale


def _cell_errors(nb):
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    errors = []
    for index, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                tb = [ansi.sub("", line).strip() for line in (out.get("traceback") or [])]
                tb = [line for line in tb if line]
                errors.append(
                    {
                        "cell": index,
                        "ename": out.get("ename"),
                        "evalue": out.get("evalue"),
                        "last_traceback_line": tb[-1] if tb else None,
                    }
                )
    return errors


def _cell_stream(nb, index):
    chunks = []
    for out in nb.cells[index].get("outputs", []):
        if out.get("output_type") == "stream":
            chunks.append(out.get("text", ""))
        elif out.get("output_type") == "execute_result":
            chunks.append(str(out.get("data", {}).get("text/plain", "")))
    return "".join(chunks)


def stage_cells():
    """Execute the three previously failing cells in isolation, plus legacy probes."""
    import nbformat

    prepare_mirror()
    out = {"cells": [], "probes": [], "errors": []}

    jobs = [
        ("compare_landscapes_corr-prestige-merit.ipynb", 10, [0, 1, 10], "cell10"),
        ("compare_landscapes_corr-prestige-merit.ipynb", 11, [0, 1, 11], "cell11"),
        ("longitudinalcomparisions.ipynb", 2, [0, 1, 2], "cell2"),
    ]

    for stem_file, cell_index, prefix_cells, tag in jobs:
        path = os.path.join(REPO, "notebooks", stem_file)
        nb = nbformat.read(path, as_version=4)
        src = [nb.cells[i] for i in prefix_cells]
        nb.cells = [nbformat.v4.new_code_cell(CAPTURE_CELL)] + src

        prefix = f"{os.path.splitext(stem_file)[0]}_{tag}"
        t0 = time.time()
        fatal, figs, stale = _execute(nb, os.path.join(FIG, "cells"), prefix)
        errors = _cell_errors(nb)
        # capture cell shifted every index by one
        for err in errors:
            err["original_cell"] = prefix_cells[err["cell"]] if err["cell"] < len(prefix_cells) else None

        target_pos = len(prefix_cells)  # index of the target cell in the executed nb
        out["cells"].append(
            {
                "notebook": f"notebooks/{stem_file}",
                "cell": cell_index,
                "source": nb.cells[target_pos].source
                if isinstance(nb.cells[target_pos].source, str)
                else "".join(nb.cells[target_pos].source),
                "fatal": fatal,
                "seconds": round(time.time() - t0, 1),
                "errors": errors,
                "stdout": _cell_stream(nb, target_pos),
                "figures": [png_info(p) for p in figs],
                "stale_figures_not_rewritten": [os.path.relpath(p, REPO) for p in stale],
                "executed_sha256": hashlib.sha256(
                    json.dumps([c.source for c in nb.cells[1:]]).encode()
                ).hexdigest(),
            }
        )
        checkpoint("cell_" + tag, out["cells"][-1])
        print(f"[{tag}] errors={len(errors)} figures={len(figs)} ({time.time() - t0:.1f}s)", flush=True)

    # legacy probes: old spellings through the shim against legacy CSVs
    probes_dir = os.path.join(FIG, "probes")
    os.makedirs(probes_dir, exist_ok=True)
    import matplotlib.pyplot as plt

    plt.close("all")
    legacy_datasets = {
        os.path.basename(p).replace("testrun_corr-prestige-merit_", "").replace(".csv", ""): live.read_csv(p)
        for p in sorted(glob.glob(os.path.join(REPO, "data", "testrun_corr-prestige-merit_*.csv")))
    }
    live.plot_smt("Mean_of_Corr_prestige_localMerit", legacy_datasets)
    path = os.path.join(probes_dir, "legacy_report_plot_smt_legacy_spelling.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")

    # numeric identity: the plotted means must equal the means of the raw legacy column
    raw_frames = [pd.read_csv(p) for p in sorted(
        glob.glob(os.path.join(REPO, "data", "testrun_corr-prestige-merit_*.csv")))]
    curiosities = sorted(set().union(*[d["initial_curiosity"].unique() for d in legacy_datasets.values()]))
    raw_plot_means = []
    for c in curiosities:
        vals = [d[d["initial_curiosity"] == c]["mean_corr_prestige_localMerit"].mean() for d in raw_frames]
        raw_plot_means.append(float(np.nanmean(vals)))
    shimmed_means = []
    for c in curiosities:
        vals = [d[d["initial_curiosity"] == c]["Mean_of_Corr_prestige_local_merit"].mean()
                for d in legacy_datasets.values()]
        shimmed_means.append(float(np.nanmean(vals)))
    assert np.allclose(raw_plot_means, shimmed_means, equal_nan=True), (raw_plot_means, shimmed_means)
    out["probes"].append(
        {
            "probe": "plot_smt(legacy camelCase spelling) on legacy report CSVs",
            "input_files": sorted(legacy_datasets),
            "figure": png_info(path),
            "curiosities": [float(c) for c in curiosities],
            "means_from_raw_legacy_column": raw_plot_means,
            "means_from_shimmed_column": shimmed_means,
            "max_abs_difference": float(np.nanmax(np.abs(np.array(raw_plot_means) - np.array(shimmed_means)))),
        }
    )
    plt.close("all")

    # per-step probe: a folder of legacy frames with curiosity in the filename
    probe_folder = os.path.join(OUT, "legacy_frame_probe")
    os.makedirs(probe_folder, exist_ok=True)
    for c in (0.1, 0.2, 0.3, 0.4, 0.5):
        src = os.path.join(REPO, "data", "model_long_random.csv")
        dst = os.path.join(probe_folder, f"probe_curiosity{c}_seed0.csv")
        if not os.path.exists(dst):
            os.symlink(src, dst)
    live.plot_all_curiosities(probe_folder)
    path = os.path.join(probes_dir, "legacy_frame_plot_all_curiosities.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    out["probes"].append(
        {
            "probe": "plot_all_curiosities on legacy frame CSVs (corr_prestige_localMerit on disk)",
            "input_folder": os.path.relpath(probe_folder, REPO),
            "figure": png_info(path),
        }
    )

    checkpoint("cells", out)
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=2))
    for entry in out["cells"]:
        print(f"  {entry['notebook']} cell {entry['cell']}: errors={len(entry['errors'])} "
              f"figures={len(entry['figures'])} nontrivial={sum(f['nontrivial'] for f in entry['figures'])}")
    return out


# --------------------------------------------------------------- stage notebooks


def stage_notebooks(only=None):
    import nbformat

    prepare_mirror()
    notebook_ckpt = os.path.join(CKPT, "notebooks")
    os.makedirs(notebook_ckpt, exist_ok=True)
    out = {"results": [], "errors": []}

    for stem_file in SIX_NOTEBOOKS:
        if only and stem_file not in only:
            continue
        path = os.path.join(REPO, "notebooks", stem_file)
        nb = nbformat.read(path, as_version=4)
        src_cells = list(nb.cells)
        nb.cells = [nbformat.v4.new_code_cell(CAPTURE_CELL)] + src_cells
        prefix = os.path.splitext(stem_file)[0]
        t0 = time.time()
        fatal, figs, stale = _execute(nb, FIG, prefix)
        errors = _cell_errors(nb)
        for err in errors:
            err["original_cell"] = err["cell"] - 1 if err["cell"] > 0 else None
            if 0 <= err["cell"] - 1 < len(src_cells):
                err["cell_source_head"] = "\n".join(src_cells[err["cell"] - 1].source.splitlines()[:4])

        art = nbformat.read(path, as_version=4)  # artifact copy without capture cell
        art.cells = list(nb.cells[1:])
        art_path = os.path.join(EXEC, stem_file)
        with open(art_path, "w", encoding="utf-8") as f:
            nbformat.write(art, f)

        result = {
            "notebook": f"notebooks/{stem_file}",
            "n_code_cells": sum(1 for c in src_cells if c.cell_type == "code"),
            "executed_clean": fatal is None and not errors,
            "fatal": fatal,
            "seconds": round(time.time() - t0, 1),
            "figures": [png_info(p) for p in figs],
            "n_figures": len(figs),
            "stale_figures_not_rewritten": [os.path.relpath(p, REPO) for p in stale],
            "errors": errors,
            "executed_path": os.path.relpath(art_path, REPO),
        }
        checkpoint("notebooks/" + prefix, result)
        out["results"].append(result)
        print(f"== {stem_file}: clean={result['executed_clean']} figures={result['n_figures']} "
              f"errors={len(errors)} ({result['seconds']}s)", flush=True)

    checkpoint("notebooks", out)
    print(json.dumps({"notebooks": len(out["results"]),
                      "clean": sum(r["executed_clean"] for r in out["results"]),
                      "figures": sum(r["n_figures"] for r in out["results"])}, indent=2))
    return out


# ------------------------------------------------------------------ stage report


def stage_report():
    base_manifest = json.load(open(os.path.join(REPO, "data", "figures_v1", "notebook_run_manifest.json")))
    base_by_nb = {r["notebook"]: r for r in base_manifest["results"]}

    new_path = os.path.join(CKPT, "notebooks.json")
    if not os.path.exists(new_path):
        per_nb = {}
        for stem_file in SIX_NOTEBOOKS:
            p = os.path.join(CKPT, "notebooks", os.path.splitext(stem_file)[0] + ".json")
            if os.path.exists(p):
                per_nb["notebooks/" + stem_file] = json.load(open(p))
        new_results = [per_nb[f"notebooks/{s}"] for s in SIX_NOTEBOOKS if f"notebooks/{s}" in per_nb]
    else:
        new_results = json.load(open(new_path))["results"]

    cells_path = os.path.join(CKPT, "cells.json")
    cells = json.load(open(cells_path)) if os.path.exists(cells_path) else {"cells": [], "probes": []}
    units = json.load(open(os.path.join(CKPT, "units.json"))) if os.path.exists(os.path.join(CKPT, "units.json")) else {}

    rows = []
    for res in new_results:
        stem = os.path.basename(res["notebook"])
        base = base_by_nb.get(res["notebook"], {})
        rows.append(
            {
                "notebook": stem,
                "code_cells": res["n_code_cells"],
                "baseline_figures": base.get("n_figures"),
                "baseline_clean": base.get("executed"),
                "baseline_errors": [e["cell"] for e in base.get("errors", [])],
                "new_figures": res["n_figures"],
                "new_clean": res["executed_clean"],
                "new_errors": [e["original_cell"] for e in res["errors"]],
                "delta_figures": (res["n_figures"] - base["n_figures"]) if base else None,
                "seconds": res["seconds"],
            }
        )

    all_pngs = sorted(glob.glob(os.path.join(FIG, "**", "*.png"), recursive=True))
    inventory = [png_info(p) for p in all_pngs]
    nontrivial = [i for i in inventory if i["nontrivial"]]

    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "units": {
            "baseline_sha256": units.get("baseline_sha256"),
            "alias_resolutions": len(units.get("alias_matrix", [])),
            "frames": units.get("frames", {}),
            "read_shim": units.get("read_shim"),
            "unknown_name_raises_KeyError": units.get("unknown_name_raises_KeyError"),
            "errors": units.get("errors"),
        },
        "cells": cells.get("cells", []),
        "probes": cells.get("probes", []),
        "regression": rows,
        "figures": {
            "gross_png": len(inventory),
            "net_nontrivial": len(nontrivial),
            "below_10kB": [i["path"] for i in inventory if not i["nontrivial"]],
            "gross_by_area": {
                "cells": len(glob.glob(os.path.join(FIG, "cells", "*.png"))),
                "probes": len(glob.glob(os.path.join(FIG, "probes", "*.png"))),
                "notebooks": len(glob.glob(os.path.join(FIG, "*.png"))),
            },
        },
        "totals": {
            "baseline_figures": base_manifest.get("figures_total"),
            "new_figures": sum(r["n_figures"] for r in new_results),
            "baseline_clean_notebooks": base_manifest.get("notebooks_net_clean"),
            "new_clean_notebooks": sum(r["executed_clean"] for r in new_results),
        },
    }
    ensure_dirs()
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({"totals": summary["totals"], "figures": summary["figures"]}, indent=2))
    print(f"{'notebook':52s} {'cells':>5s} {'base':>5s} {'new':>4s} {'d':>3s} clean(base->new) errors(new)")
    for r in rows:
        print(f"{r['notebook']:52s} {r['code_cells']:5d} {r['baseline_figures']!s:>5s} {r['new_figures']:4d} "
              f"{r['delta_figures']!s:>3s} {str(r['baseline_clean']):>5s}->{str(r['new_clean']):5s} "
              f"{r['new_errors']}")
    for entry in summary["cells"]:
        print(f"cell {entry['cell']:2d} {os.path.basename(entry['notebook']):46s} "
              f"errors={len(entry['errors'])} figures={len(entry['figures'])}")
    return summary


STAGES = {"units": stage_units, "cells": stage_cells, "notebooks": stage_notebooks, "report": stage_report}


def main(argv):
    if not argv or argv[0] not in STAGES:
        print(__doc__)
        return 2
    ensure_dirs()
    stage = argv[0]
    if stage == "notebooks" and len(argv) > 1:
        STAGES[stage](only=argv[1:])
    else:
        STAGES[stage]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
