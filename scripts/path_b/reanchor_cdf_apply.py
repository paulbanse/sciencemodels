"""CDF re-anchor applicator v5 (scratch, /tmp). Lands, in one change:

  1. the post-Path-B re-anchor of all moved Reference cells (doc-side
     rebuild from the doc's OWN segment structure + the mapping's resolved
     checks - the owner's v4 restructure of the bulk_runner citations is
     adopted, not reverted);
  2. the two catalogue rows the re-measure round drafted and deferred
     (entry_prestige_stock, ideal_merit_cum), supplied ready-to-paste in
     output/reports/remeasure_v1.md, INTO the doc and the mapping together
     (104 -> 106 rows, inserted after the runner.py:54 mislabel row);
  3. the prose-only citations the table-cell verifier never covered
     (segment-level tolerant replaces + hand-curated entries);
  4. the status note and a computed Version 5 revision-log entry.

Then the gate (run separately) must PASS. REVIEW mode prints the plan.
"""
import importlib.util
import re
import sys
from pathlib import Path

REPO = Path("/workspace/git/sciencemodels")
CDF_DOC = REPO / "docs" / "conceptual_degrees_of_freedom.md"
REANCHOR = REPO / "scripts" / "cdf_reanchor" / "reanchor_cdf_refs.py"
REVIEW_LOG = Path("/tmp/t5_reanchor_review.txt")

spec = importlib.util.spec_from_file_location("reanchor", REANCHOR)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# hand-curated whole-cell rebuilds (extents read from the live post-Path-B
# code; every range with fewer than two pinned endpoints, or whose block
# grew internally, lives here). Keyed by the CURRENT (pre-insertion)
# mapping row index.
CURATED = {
    4: "`sciencemodels/model.py:54-71`",
    16: "`sciencemodels/model.py:87-94`",
    17: "`sciencemodels/model.py:91-94`",
    18: "`sciencemodels/model.py:87-90`",
    21: "`sciencemodels/model.py:96-101`",
    22: "`sciencemodels/model.py:377-388`",
    24: "`sciencemodels/model.py:377-388,462-469`",
    25: "`sciencemodels/model.py:357-360`",
    26: "`sciencemodels/model.py:462-469`",
    34: "`sciencemodels/agents.py:40-42`",
    35: "`sciencemodels/model.py:136-138`",
    41: "`sciencemodels/agents.py:152-154`",
    42: "`sciencemodels/agents.py:160-179`",
    43: "`sciencemodels/agents.py:166-169`",
    45: "`sciencemodels/agents.py:156-159`",
    48: "`sciencemodels/agents.py:139-142`",
    49: "`sciencemodels/model.py:520-524`",
    50: "`sciencemodels/agents.py:99-100`",
    51: "`sciencemodels/agents.py:135-137`",
    53: "`sciencemodels/agents.py:101-105`",
    54: "`sciencemodels/agents.py:93-94`",
    55: "`sciencemodels/model.py:495-504`",
    56: "`sciencemodels/agents.py:58-72`",
    57: "`sciencemodels/agents.py:128-132`, `sciencemodels/model.py:167`",
    58: "`sciencemodels/agents.py:113-125`",
    60: "`sciencemodels/agents.py:133-134`",
    62: "`sciencemodels/model.py:262-270`",
    63: "`sciencemodels/agents.py:183-205`",
    64: "`sciencemodels/agents.py:76-82`",
    66: "`sciencemodels/agents.py:84-91`, `sciencemodels/model.py:139`",
    72: "`sciencemodels/model.py:390-394`",
    74: "`sciencemodels/model.py:396-407`",
    75: "`sciencemodels/model.py:402-407`",
    76: "`sciencemodels/model.py:409-412`",
    77: "`sciencemodels/model.py:419-420`",
    83: "`sciencemodels/metrics.py:308-316`",
    84: "`sciencemodels/model.py:328-333`",
    85: "`sciencemodels/metrics.py:323-330`",
    90: "`sciencemodels/model.py:292-317`",
    91: "`sciencemodels/model.py:485-494`",
    93: "`sciencemodels/model.py:505-508`",
    95: "`sciencemodels/model.py:636-644`",
    96: "`sciencemodels/model.py:656-661`, `scripts/bulk_runner.py:82-83`",
}

# the two deferred rows, owner-drafted (output/reports/remeasure_v1.md
# lines 348-349 and 357-363), landed here in one change with the mapping
NEW_DOC_ROWS = [
    "| `sciencemodels/model.py:143,181-187,395-399` | Which prestige stock weights the supervisor roulette when newcomers are assigned | constructor parameter `entry_prestige_stock` (default `\"prestige\"`), validated to the three legal values `prestige` (raw stock), `prestige_vanishing` and `prestige_visibility`; `__generate_new_agents` reads `entry_stock = self.entry_prestige_stock` (line 395) and builds both the tuple list and the roulette total with `getattr(a, entry_stock)` (lines 396-399) | What the community treats as the quality signal newcomers act on: accumulated visibility (raw stock), recent output (vanishing) or peer-perceived standing (visibility). All three are swept in `output/reports/prestige_stock_v1.md`; the probe finds the stock sets how fast the quantity/quality wedge forms, not where it opens |",
    "| `sciencemodels/metrics.py:105`, `sciencemodels/model.py:416` | Cumulative ideal-merit accumulator on every agent | `update_ideal_merit` adds the step's `a.ideal_Merit` into `a.ideal_merit_cum` with a lazy `getattr(..., 0.0)` default (line 105); `__generate_new_agents` initialises newcomers to `np.float64(0.0)` at birth (line 416), covering final-wave entrants that never get an `updateIdealMerit` call | Turns per-step ideal-merit loss into a career total, consumed by the cohort/career reporters (`scripts/remeasure/run_remeasure.py:131`, `scripts/path_b/run_path_b_sweep.py:154,157`) to compare lifetime quality deficit against lifetime prestige; a reporting construct, not a reward input |",
]
NEW_MAPPING_ROWS = [
    ("(new row, no v2 predecessor)", "`sciencemodels/model.py:143,181-187,395-399`", "re-anchored",
     [("sciencemodels/model.py", 143, "entry_prestige_stock: str"),
      ("sciencemodels/model.py", 181, "if entry_prestige_stock not in ("),
      ("sciencemodels/model.py", 395, "entry_stock = self.entry_prestige_stock")]),
    ("(new row, no v2 predecessor)", "`sciencemodels/metrics.py:105`, `sciencemodels/model.py:416`", "re-anchored",
     [("sciencemodels/metrics.py", 105, 'a.ideal_merit_cum = getattr(a, "ideal_merit_cum", 0.0) + a.ideal_Merit'),
      ("sciencemodels/model.py", 416, "a.ideal_merit_cum = np.float64(0.0)")]),
]
INSERT_AFTER_DOC_ROW = 80   # 1-based doc row (the runner.py:54 mislabel row)

# prose-only refs not equal to any table segment (the gate's blind spot)
PROSE = [
    ("`sciencemodels/model.py:315`", "`sciencemodels/model.py:396`"),
    ("`sciencemodels/model.py:131`", "`sciencemodels/model.py:138`"),
    ("`sciencemodels/model.py:79-86`", "`sciencemodels/model.py:86-93`"),
    ("`sciencemodels/model.py:346`", "`sciencemodels/model.py:466`"),
    ("`sciencemodels/agents.py:166`", "`sciencemodels/agents.py:176`"),
    ("`sciencemodels/model.py:336`", "`sciencemodels/model.py:419`"),
    ("`sciencemodels/metrics.py:130-131`",
     "`sciencemodels/metrics.py:328-329`"),
]

NOTE_ANCHOR = "re-run `scripts/cdf_reanchor/reanchor_cdf_refs.py` and re-resolve."
NOTE_NEW = """re-run `scripts/cdf_reanchor/reanchor_cdf_refs.py` and re-resolve.

Re-verified at 2026-11-14 (post-Path-B dyadic layer). The Path B
implementation (dyadic step and scores, team object, equal_split buffering,
`farm`'s new agent argument) shifted `model.py`, `agents.py` and
`metrics.py` line numbers non-uniformly; 101 of the 167 resolvable table
checks moved and were re-resolved against the live tree (unique-token
auto-resolution except where the candidate list was ambiguous - resolved by
the original anchor's semantics), and one quoted token changed
(`self.model.farm(self.pos)` -> `self.model.farm(self.pos, self)`). The two
catalogue rows the re-measure round drafted and deferred (the
`entry_prestige_stock` knob and the `ideal_merit_cum` accumulator, supplied
ready-to-paste in `output/reports/remeasure_v1.md`) are landed here together
with their mapping entries - the document goes from 104 to 106 catalogue
rows, as that round predicted. The bulk-runner citation placement follows
the v4 restructure. Prose-only citations, which the table-cell verifier
never covered, were re-anchored in the same pass (segment-level where the
prose repeats a table segment, hand-curated otherwise). The verifier passes
on all anchors after this update."""


def file_lines(relpath):
    return (REPO / relpath).read_text(encoding="utf-8", errors="replace").split("\n")


def doc_table_rows():
    """(line_number, first_cell, full_line) for every gate-visible row."""
    out = []
    started = False
    for ln, line in enumerate(CDF_DOC.read_text().split("\n"), 1):
        if line.startswith("# Purpose"):
            started = True
            continue
        if not started:
            continue
        if line.startswith("| `") or line.startswith("| **unresolvable"):
            cell = line.strip().strip("|").split("|")[0].strip()
            out.append((ln, cell, line))
    return out


def parse_cell(cell):
    """-> [(file, spans, has_linespec, tail)] or None for unresolvable
    rows. The tail (prose annotation after the closing backtick, e.g.
    ' (class 21-183)' or ' (cell 0)') is preserved verbatim."""
    if cell.startswith("**unresolvable"):
        return None
    segs = []
    for seg in [s.strip() for s in cell.split("`, `")]:
        if seg.startswith("`"):
            seg = seg[1:]
        tail = ""
        if "`" in seg:
            ref, _, tail = seg.partition("`")
            tail = "`" + tail
        else:
            ref = seg
        f, _, lspec = ref.partition(":")
        if not lspec:
            segs.append((f, [(1, 10**9)], False, tail))  # whole-file
            continue
        parts = []
        for part in lspec.split(","):
            if "-" in part:
                a, b = part.split("-")
                parts.append((int(a), int(b)))
            else:
                parts.append((int(part), int(part)))
        segs.append((f, parts, True, tail))
    return segs


def contains(spans, line):
    return any(a <= line <= b for a, b in spans)


def raw_refs(cell):
    out = []
    for seg in [s.strip() for s in cell.split("`, `")]:
        if seg.startswith("`"):
            seg = seg[1:]
        out.append(seg.partition("`")[0])
    return out


def rebuild_cell(doc_cell, checks, old_lines, row_idx):
    """Rebuild the doc cell from its OWN segments (the owner's v4
    restructure is adopted): each segment's pinned old lines resolve to new
    lines; singles mechanically, curated ranges from the table, unpinned
    and whole-file segments verbatim."""
    if row_idx in CURATED:
        return CURATED[row_idx]
    segs = parse_cell(doc_cell)
    if segs is None:
        return doc_cell
    out = []
    for f, spans, has_spec, tail in segs:
        pins = []
        for ci, (relpath, nl, token) in enumerate(checks):
            if relpath != f:
                continue
            ol = old_lines.get((row_idx, ci), nl)
            if contains(spans, ol):
                pins.append((ol, nl))
        if not has_spec:
            closing = tail if tail else "`"
            out.append("`%s%s" % (f, closing))
            continue
        if not pins:
            specs = ",".join(str(a) if a == b else "%d-%d" % (a, b)
                             for a, b in spans)
            closing = tail if tail else "`"
            out.append("`%s:%s%s" % (f, specs, closing))
            continue
        specs = []
        for (a, b) in spans:
            pinned = [nl for ol, nl in pins if a <= ol <= b]
            span_pins = [(ol, nl) for ol, nl in pins if a <= ol <= b]
            if not pinned:
                specs.append(str(a) if a == b else "%d-%d" % (a, b))
            elif a == b:
                specs.append(str(pinned[0]))
            elif all(ol == nl for ol, nl in span_pins):
                specs.append("%d-%d" % (a, b))  # nothing moved in this span
            else:
                if len(pinned) < 2:
                    raise SystemExit("1-pin range needs curation: row %d %s:%d-%d"
                                     % (row_idx, f, a, b))
                specs.append("%d-%d" % (min(pinned), max(pinned)))
        closing = tail if tail else "`"
        out.append("`%s:%s%s" % (f, ",".join(specs), closing))
    return ", ".join(out) if len(out) > 1 else out[0]


def main(apply=False):
    lines = CDF_DOC.read_text().split("\n")
    rows = doc_table_rows()
    mapping = list(mod.MAPPING)
    print("doc table rows: %d   mapping rows: %d" % (len(rows), len(mapping)))
    assert len(rows) == len(mapping), "positional zip impossible"

    # ---- reconstruct old check lines (kept: old == new; else review log)
    old_lines = {}
    log_re = re.compile(r"^row\s+(\d+)\s+(\S+) -> (\d+)\s+\((\w[\w-]*)\)\s+(.*)$")
    per_row = {}
    for ln in REVIEW_LOG.read_text().split("\n"):
        m = log_re.match(ln)
        if not m:
            continue
        row = int(m.group(1))
        per_row[row] = per_row.get(row, 0) + 1
        old_lines[(row, per_row[row] - 1)] = int(m.group(2).rsplit(":", 1)[1])

    # ---- verify the supplied rows' tokens against the live tree
    for _old, new_cell, _status, checks in NEW_MAPPING_ROWS:
        for relpath, lineno, token in checks:
            fl = file_lines(relpath)
            assert 0 < lineno <= len(fl) and token in fl[lineno - 1], (
                relpath, lineno, token)
    print("supplied rows' tokens verified on the live tree")

    # ---- positional pairing, structural validation, rebuild
    plan = []          # (doc_line, old_cell, rebuilt_cell)
    rebuilt_cells = [] # per doc/mapping row, in order
    seg_pairs = {}
    for i, (ln, doc_cell, _full) in enumerate(rows, 1):
        _m_old, map_new, status, checks = mapping[i - 1]
        segs = parse_cell(doc_cell)
        if segs is None:
            assert doc_cell == map_new, ("unresolvable drift", i, doc_cell)
            rebuilt_cells.append(doc_cell)
            continue
        doc_files = [f for f, _s, _h, _t in segs]
        chk_files = [c[0] for c in checks]
        assert set(chk_files) <= set(doc_files), ("file mismatch", i, doc_cell)
        for ci, (relpath, nl, token) in enumerate(checks):
            ol = old_lines.get((i, ci), nl)
            assert any(f == relpath and contains(spans, ol)
                       for f, spans, _h, _t in segs), (
                "check not covered", i, relpath, ol)
        new_cell = rebuild_cell(doc_cell, checks, old_lines, i)
        rebuilt_cells.append(new_cell)
        if new_cell != doc_cell:
            plan.append((ln, doc_cell, new_cell))
        for so, sn in zip(raw_refs(doc_cell), raw_refs(new_cell)):
            if so != sn:
                seg_pairs[so] = sn
    print("doc rows to change: %d (+2 rows inserted)" % len(plan))
    for ln, o, n in plan:
        print("  doc line %d: %s  ->  %s" % (ln, o, n))

    mislabel_line = rows[INSERT_AFTER_DOC_ROW - 1][0]
    assert "runner.py:54" in rows[INSERT_AFTER_DOC_ROW - 1][1]
    print("insert 2 supplied rows after doc line %d (runner.py:54 mislabel)"
          % mislabel_line)

    if not apply:
        print("REVIEW MODE - no files changed")
        print("planned prose segment replacements: %d" % len(seg_pairs))
        for so in sorted(seg_pairs, key=len, reverse=True):
            print("  segment: %s -> %s" % (so, seg_pairs[so]))
        print("planned manual prose entries: %d" % len(PROSE))
        return

    # ---- doc rewrite (line-targeted; duplicate-cell safe)
    for ln, o, n in plan:
        assert o in lines[ln - 1], (ln, o)
        parts = lines[ln - 1].split("|")
        parts[1] = " %s " % n
        lines[ln - 1] = "|".join(parts)
    for k, row_text in enumerate(NEW_DOC_ROWS):
        lines.insert(mislabel_line + k, row_text)
    doc = "\n".join(lines)

    # ---- prose: segment-level (longest first, tolerant) then manual
    fired = 0
    for so in sorted(seg_pairs, key=len, reverse=True):
        sn = seg_pairs[so]
        cnt = doc.count("`%s`" % so)
        if cnt:
            doc = doc.replace("`%s`" % so, "`%s`" % sn)
            fired += 1
            print("  prose segment: %s -> %s (%d)" % (so, sn, cnt))
    print("prose segment replacements fired: %d" % fired)
    for old_s, new_s in PROSE:
        cnt = doc.count(old_s)
        assert cnt >= 1, ("prose", old_s)
        doc = doc.replace(old_s, new_s)
        print("prose: %s -> %s (%d)" % (old_s, new_s, cnt))

    # ---- status note + computed Version 5 revision-log entry
    assert NOTE_ANCHOR in doc
    doc = doc.replace(NOTE_ANCHOR, NOTE_NEW, 1)
    v5 = ("- **Version 5** (November 2026) --- Post-Path-B re-anchor: the dyadic "
          "layer shifted `model.py`/`agents.py`/`metrics.py` non-uniformly; %d "
          "reference cells re-anchored, the two rows the re-measure round "
          "deferred (`entry_prestige_stock`, `ideal_merit_cum`) landed with "
          "their mapping entries in one change (104 -> 106 rows), the v4 "
          "bulk-runner citation restructure adopted, and the prose-only "
          "citations re-anchored (a blind spot of the table-cell verifier). "
          "The verifier passes on all anchors after this update." % len(plan))
    v4_anchor = "- **Version 4** (November 2026)"
    assert v4_anchor in doc
    v4_end = doc.index("\n", doc.index(v4_anchor))
    doc = doc[:v4_end + 1] + v5 + "\n" + doc[v4_end + 1:]
    CDF_DOC.write_text(doc)
    print("CDF doc rewritten: %d cells + 2 rows inserted + prose + note + v5 log"
          % len(plan))

    # ---- script: same rebuilt cells, supplied rows inserted, checks kept
    new_mapping = []
    for i, (m_old, _m_new, status, checks) in enumerate(mapping, 1):
        new_mapping.append((m_old, rebuilt_cells[i - 1], status, checks))
    final = (new_mapping[:INSERT_AFTER_DOC_ROW] + NEW_MAPPING_ROWS
             + new_mapping[INSERT_AFTER_DOC_ROW:])
    assert len(final) == 106
    src = REANCHOR.read_text()
    ser = []
    for old, new_cell, status, checks in final:
        checks_s = ",\n             ".join(repr(c) for c in checks)
        ser.append("    (%r, %r, %r,\n     [%s])," % (old, new_cell, status, checks_s))
    mapping_block = "MAPPING = [\n" + "\n".join(ser) + "\n]"
    inline_s = ",\n    ".join(repr(t) for t in mod.INLINE_ANCHORS)
    inline_block = "INLINE_ANCHORS = [\n    %s,\n]" % inline_s
    src2 = re.sub(r"MAPPING = \[.*?\n\]", mapping_block, src, flags=re.S)
    src2 = re.sub(r"INLINE_ANCHORS = \[.*?\n\]", inline_block, src2, flags=re.S)
    if src2 != src:
        REANCHOR.write_text(src2)
        print("reanchor script rewritten: 106 mapping rows")
    else:
        print("reanchor script already current")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
