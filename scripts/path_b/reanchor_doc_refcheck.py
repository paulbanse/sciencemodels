"""Cross-check every code reference in the CDF doc against the (post-apply)
reanchor mapping + inline anchors. Classifies each ref as:
  covered-mapping  - equals some mapping row's OLD or NEW cell (table rows)
  covered-inline   - file:line matches an inline anchor's old or new line
  untracked        - neither: a prose citation the gate never verifies
For untracked refs, verifies the token-on-cited-line against the live tree
where the line range is small enough to eyeball, and prints the cited lines.
"""
import importlib.util
import re
from pathlib import Path

REPO = Path("/workspace/git/sciencemodels")
DOC = REPO / "docs" / "conceptual_degrees_of_freedom.md"
REANCHOR = REPO / "scripts" / "cdf_reanchor" / "reanchor_cdf_refs.py"

spec = importlib.util.spec_from_file_location("reanchor", REANCHOR)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

doc = DOC.read_text()

# every backticked file:line ref in the doc (prose and table alike)
ref_re = re.compile(r"`((?:sciencemodels|scripts|notebooks)/[A-Za-z0-9_ .\-]+\.(?:py|ipynb)):([0-9,\- ]+)`")
refs = ref_re.findall(doc)
print("total backticked file:line refs in doc:", len(refs))

# mapping cells (old v3.0 cells are gone from the script; the script now
# holds the NEW cells) + inline anchors
map_cells = set()
for old, new, status, checks in mod.MAPPING:
    map_cells.add(new)
inline = {(f, l) for f, l, t in mod.INLINE_ANCHORS}

def ref_matches_cell(ref, cell):
    # ref = (file, linetext); cell like 'sciencemodels/model.py:128,160'
    m = re.match(r"^([^:]+):(.+)$", cell)
    if not m:
        return False
    cf, cl = m.group(1), m.group(2)
    return ref[0] == cf and ref[1].replace(" ", "") == cl.replace(" ", "")

untracked = []
for i, (f, lt) in enumerate(refs):
    if any(ref_matches_cell((f, lt), c) for c in map_cells):
        continue
    # inline: single line or small range containing an anchor line
    nums = []
    for part in lt.split(","):
        if "-" in part:
            a, b = part.split("-")
            nums.extend(range(int(a), int(b) + 1))
        else:
            nums.append(int(part))
    if any((f, n) in inline for n in nums):
        continue
    untracked.append((f, lt, nums))

print("covered by mapping cells or inline anchors: %d" % (len(refs) - len(untracked)))
print("UNTRACKED prose refs (gate blind spot): %d" % len(untracked))
for f, lt, nums in untracked:
    lines = (REPO / f).read_text(encoding="utf-8", errors="replace").split("\n")
    print("  %s:%s" % (f, lt))
    for n in nums:
        if 0 < n <= len(lines):
            print("      %4d| %s" % (n, lines[n - 1]))
        else:
            print("      %4d| <OUT OF RANGE - file has %d lines>" % (n, len(lines)))
