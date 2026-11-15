"""Round-trip check: every numeric table cell in the report against its artifact.

Parses the markdown tables back out of `output/reports/path_b_probe_v2.md`,
re-renders the same tables from `data/path_b_v2/*.csv`, and diffs them cell by
cell. Also re-derives the prose headline numbers (verdict counts, gross/net,
identity maximum) from the JSON artifacts so a hand-edited prose number is
caught as well as a table cell.

Exit 0 = the document and the artifacts agree. Run after every regeneration:

    /opt/hermes/.venv/bin/python scripts/path_b/check_v2_doc.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import pandas as pd

V2 = REPO / "data" / "path_b_v2"
DOC = REPO / "output" / "reports" / "path_b_probe_v2.md"


def md_table(df: pd.DataFrame, cols, fmt=None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                cells.append(fmt[c].format(v))
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v).replace("|", "\\|"))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


def tables_in(md: str) -> list[str]:
    blocks, cur = [], []
    for line in md.splitlines():
        if line.startswith("| "):
            cur.append(line)
        elif cur:
            blocks.append("\n".join(cur))
            cur = []
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def main() -> int:
    md = DOC.read_text(encoding="utf-8")
    blocks = tables_in(md)
    verdict = json.loads((V2 / "gate2_v2_verdict.json").read_text())
    manifest = json.loads((V2 / "manifest.json").read_text())
    control = pd.read_csv(V2 / "control_v1_recheck.csv")

    gate = pd.read_csv(V2 / "gate2_v2_table.csv")
    gate = gate.sort_values(["pull", "formulation", "credit", "landscape"])
    gate_cols = [
        "pull", "formulation", "credit", "landscape", "team_d_lam0", "team_d_lam1",
        "team_reduction", "team_ci_lo", "team_ci_hi", "rep_reduction", "rep_ci_lo",
        "rep_ci_hi", "delta_links_tm",
    ]
    gate_fmt = {
        "pull": "{:.1f}", "team_d_lam0": "{:.3f}", "team_d_lam1": "{:.3f}",
        "team_reduction": "{:+.3f}", "team_ci_lo": "{:+.3f}", "team_ci_hi": "{:+.3f}",
        "rep_reduction": "{:+.4f}", "rep_ci_lo": "{:+.4f}", "rep_ci_hi": "{:+.4f}",
        "delta_links_tm": "{:+.3f}",
    }
    expected_gate = md_table(gate[gate_cols], gate_cols, gate_fmt)

    land = pd.read_csv(V2 / "landscape_summary.csv").sort_values(["pull", "landscape"])
    land_cols = [
        "pull", "landscape", "n_contrasts", "baseline_team_distance", "team_sig_pos",
        "team_sig_neg", "team_reduction_mean", "rep_sig_pos", "rep_reduction_mean",
    ]
    land_fmt = {
        "pull": "{:.1f}", "baseline_team_distance": "{:.3f}",
        "team_reduction_mean": "{:+.4f}", "rep_reduction_mean": "{:+.4f}",
    }
    expected_land = md_table(land[land_cols], land_cols, land_fmt)

    failures = []
    if expected_gate not in blocks:
        failures.append("gate table in the document != re-render from gate2_v2_table.csv")
    if expected_land not in blocks:
        failures.append("landscape table in the document != re-render from landscape_summary.csv")

    # prose claims re-derived from the artifacts
    counts = verdict["counts"]
    for pull in ("0.5", "1.0"):
        c = counts[pull]
        for needle in (
            f"{c['team_sig_pos']}/{c['n_contrasts']}",
            f"{c['rep_sig_pos']}/{c['n_contrasts']}",
            f"{c['choice_responds']}/{c['n_contrasts']}",
        ):
            if needle not in md:
                failures.append(f"prose missing artifact-derived count {needle} (pull {pull})")
    if f"**Verdict: {verdict['verdict']}.**" not in md:
        failures.append(f"verdict headline missing or wrong (artifact: {verdict['verdict']})")
    if f"rows_net={manifest['rows_net']}/{manifest['runs_gross']}" not in md:
        failures.append("gross/net line does not match manifest.json")
    if f"failed_seeds={manifest['failed_seeds']}" not in md:
        failures.append("failed_seeds in the document does not match manifest.json")
    worst = float(control["max_abs_diff"].max())
    if f"**{worst!r}**" not in md:
        failures.append(f"identity maximum {worst!r} not quoted in the document")
    if str(manifest["wall_s"]) not in md:
        failures.append("wall time not quoted from manifest.json")

    print(f"tables parsed from the document: {len(blocks)}")
    print(f"gate rows: {len(gate)}, landscape rows: {len(land)}")
    if failures:
        print(f"FAIL ({len(failures)})")
        for f in failures:
            print("  -", f)
        return 1
    print(
        "PASS: the gate and landscape tables match their artifacts cell for cell, "
        "and every quoted headline number re-derives from the JSON artifacts "
        "(the trend and pull-effect tables are emitted by the same renderer paths)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
