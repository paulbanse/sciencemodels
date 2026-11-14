"""Classify the CDF doc's code refs into table cells (replaced via the
positional zip with the reanchor mapping) vs prose refs. Prose refs that
match no table cell string and no new mapping cell need manual PROSE
entries - the gate never verifies them (v3.0 blind spot, now closed).
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
lines = doc.split("\n")

ref_re = re.compile(r"`((?:sciencemodels|scripts|notebooks)/[A-Za-z0-9_ .\-]+\.(?:py|ipynb)):([0-9,\- ]+)`")

table_refs = []   # (row_idx_in_table_order, ref_string)
prose_refs = []   # (line_no, ref_string)
in_table = False
for ln, line in enumerate(lines, 1):
    if line.strip().startswith("|"):
        in_table = True
        for m in ref_re.finditer(line):
            table_refs.append((ln, m.group(0)))
    else:
        if in_table and line.strip() == "":
            in_table = False
        for m in ref_re.finditer(line):
            prose_refs.append((ln, m.group(0)))

print("table refs: %d   prose refs: %d" % (len(table_refs), len(prose_refs)))

new_cells = set()
for old, new, status, checks in mod.MAPPING:
    new_cells.add(new)

table_strings = set(t[1] for t in table_refs)
print("distinct table cell strings: %d" % len(table_strings))

# mapping rows with code-ref cells, in order (for the zip)
code_rows = [(i, new) for i, (old, new, status, checks) in enumerate(mod.MAPPING, 1)
             if re.search(r"\.(py|ipynb):", new)]
print("mapping rows with code-ref cells: %d (mapping total %d)" % (
    len(code_rows), len(mod.MAPPING)))

print("\nPROSE refs not covered by any table cell string and not already new:")
need = []
for ln, ref in prose_refs:
    if ref in table_strings:
        continue  # the table replacement hits all occurrences
    if ref in new_cells:
        continue  # already current
    need.append((ln, ref))
for ln, ref in need:
    print("  line %d: %s" % (ln, ref))
print("total needing manual PROSE entries: %d" % len(need))

# also verify the zip alignment: distinct table code-ref strings in doc order
# vs mapping code cells in order
doc_code_cells = []
seen_lines = set()
for ln, ref in table_refs:
    doc_code_cells.append(ref)
print("\nzip check: doc table code refs=%d, mapping code cells=%d" % (
    len(doc_code_cells), len(code_rows)))
mismatch = 0
for k, ((dln, dref), (mri, mnew)) in enumerate(zip(doc_code_cells, code_rows)):
    if dref != mnew[1]:
        mismatch += 1
        if mismatch <= 10:
            print("  POSITIONAL DIFF at %d: doc=%s  mapping(row %d)=%s" % (
                k, dref, mri, mnew[1]))
print("positional diffs (doc table vs new mapping): %d" % mismatch)
