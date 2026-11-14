"""Pair the CDF doc's 106 Reference cells with the reanchor mapping's 104
rows (order-preserving, content-validated) to locate the 2 doc rows the
v3.0 mapping never tracked.

Old check lines per mapping row are reconstructed: kept checks have
old == new (the rewritten script's check lines); non-kept checks come from
the review log (/tmp/t5_reanchor_review.txt). A doc cell pairs with a
mapping row iff files match and every check's old line falls inside the
doc cell's cited specs. Unresolvable rows pair by status.
"""
import importlib.util
import re
from pathlib import Path

REPO = Path("/workspace/git/sciencemodels")
DOC = REPO / "docs" / "conceptual_degrees_of_freedom.md"
REANCHOR = REPO / "scripts" / "cdf_reanchor" / "reanchor_cdf_refs.py"
REVIEW_LOG = Path("/tmp/t5_reanchor_review.txt")

spec = importlib.util.spec_from_file_location("reanchor", REANCHOR)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# ---- doc cells, mirroring the gate's _doc_cells() exactly
cells = []
started = False
for line in DOC.read_text().split("\n"):
    if line.startswith("# Purpose"):
        started = True
        continue
    if not started:
        continue
    if line.startswith("| `") or line.startswith("| **unresolvable"):
        cells.append(line.strip().strip("|").split("|")[0].strip())
print("doc cells:", len(cells))

# ---- reconstruct old check lines per mapping row
old_lines = {}  # (row, check_idx) -> old_line
log_re = re.compile(
    r"^row\s+(\d+)\s+(\S+) -> (\d+)\s+\((\w[\w-]*)\)\s+(.*)$")
# review log lines: 'row   4  sciencemodels/model.py:47 -> 54  (curated)  'token''
# checks within a row appear in order; count per row to recover the index
per_row_count = {}
for ln in REVIEW_LOG.read_text().split("\n"):
    m = log_re.match(ln)
    if not m:
        continue
    row = int(m.group(1))
    per_row_count[row] = per_row_count.get(row, 0) + 1
    old_lines[(row, per_row_count[row] - 1)] = int(m.group(2).rsplit(":", 1)[1])
print("review-log resolutions:", len(old_lines))

# ---- parse a cell string into [(file, [(fmt, nums)])]
def parse_cell(cell):
    if cell.startswith("**unresolvable"):
        return None
    segs = []
    for seg in [s.strip() for s in cell.split("`, `")]:
        if seg.startswith("`"):
            seg = seg[1:]
        if "`" in seg:
            ref = seg.partition("`")[0]
        else:
            ref = seg
        f, _, lspec = ref.partition(":")
        parts = []
        if not lspec:
            parts = [(1, 10**9)]  # whole-file ref, no line spec
        else:
            for part in lspec.split(","):
                if "-" in part:
                    a, b = part.split("-")
                    parts.append((int(a), int(b)))
                else:
                    parts.append((int(part), int(part)))
        segs.append((f, parts))
    return segs

def cell_contains(segs, relpath, line):
    for f, spans in segs:
        if f == relpath:
            if line == 0:  # whole-file check (single-line JSON notebooks)
                return True
            if any(a <= line <= b for a, b in spans):
                return True
    return False

# ---- pair doc cells with mapping rows (order-preserving greedy)
mapping = list(mod.MAPPING)
pairs = []          # (doc_idx, mapping_idx)
extras = []         # doc_idx unmatched
mi = 0
for di, cell in enumerate(cells):
    segs = parse_cell(cell)
    matched = None
    if mi < len(mapping):
        old, new, status, checks = mapping[mi]
        if segs is None and status == "unresolvable":
            matched = mi
        elif segs is not None:
            ok = True
            for ci, (relpath, nl, token) in enumerate(checks):
                ol = old_lines.get((mi + 1, ci), nl)  # kept: old == new
                if not cell_contains(segs, relpath, ol):
                    ok = False
                    break
            if ok:
                matched = mi
    if matched is not None:
        pairs.append((di, matched))
        mi += 1
    else:
        extras.append(di)

print("paired:", len(pairs), " extras:", len(extras), " mapping consumed:", mi, "/", len(mapping))
for di in extras:
    print("\nEXTRA doc row (1-based %d):" % (di + 1))
    print("  cell:", cells[di])
    # print the full doc table row for context
    n = 0
    for line in DOC.read_text().split("\n"):
        if line.strip().startswith("| `") or line.strip().startswith("| **unresolvable"):
            n += 1
            if n == di + 1:
                print("  full row:", line.strip()[:400])
                break
# sanity: pairing must be strictly monotone and consume all mapping rows
assert mi == len(mapping), "mapping rows left unpaired: %d" % (len(mapping) - mi)
mono = all(pairs[i][1] < pairs[i + 1][1] for i in range(len(pairs) - 1))
print("monotone pairing:", mono)
