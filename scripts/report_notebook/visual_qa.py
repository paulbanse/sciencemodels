"""Shared hardened visual QA for the section figures (Team 2, v3 round).

Two contracts, both evaluated on the RENDERED canvas (not on the sidecar and
not on text-vs-text boxes alone, which is how the F12 overlay bug and the
F9/F13 text-vs-title collisions slipped through the v2 verifiers):

1. **Rendered-artist data contract** -- per panel:
   - the number of plotted series / points / bars matches the panel's
     single assigned metric (catches unfiltered multi-metric overlays, the
     F12 catastrophe);
   - the min/max of every plotted value, including error-bar endpoints,
     matches the ``*_data.csv`` rows for that panel's metric at ``tol``;
   - every plotted value lies strictly inside the axes ``ylim`` with at
     least ``headroom`` (default 3%) vertical clearance (catches clipped
     bars such as F9_v2 panel B).

2. **Collision-free layout contract** -- for all visible text artists
   (titles, suptitle, axis labels, tick labels, ``ax.texts``, legend boxes
   AND legend texts, ``fig.texts``):
   - zero clipping by the canvas;
   - zero pairwise overlap, including text-vs-title;
   - zero overlap with plotted data: lines (sampled segments), bar
     rectangles, error-bar segments and caps, markers, and the spines;
   - zero duplicate legend labels in any legend.

Usage (inside a builder, BEFORE ``fig.savefig``)::

    import visual_qa
    report = visual_qa.assert_figure_clean(fig, [
        {"ax": ax_a, "metric": "harvested knowledge share",
         "expect_lines": 6, "points_per_line": 3,
         "sidecar": sidecar},
        ...
    ])

The spec dicts accept:
    ax                 (Axes)                      required
    metric             (str)     sidecar metric name (for the data contract)
    expect_lines       (int)     number of data series (errorbar/line artists)
    points_per_line    (int)     finite points per series
    expect_bars        (int)     number of bars
    sidecar            (DataFrame) full sidecar; filtered to ``metric``
    filters            (dict)    extra column filters applied to the sidecar
    value_col          (str)     "value" (default)
    ci_cols            ((lo, hi) names) default ("ci_lo", "ci_hi")
    skip_headroom      (bool)    exempt (rare; e.g. a deliberately clipped axis)
    legend_over_data_ok(bool)    allow legend over data on this panel
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PIXEL_TOL = 0.75          # allowed bbox overlap in pixels (anti-aliasing)
CANVAS_TOL = 1.0          # allowed canvas overshoot in pixels
DEFAULT_HEADROOM = 0.03


@dataclass
class Failures:
    items: list[str] = field(default_factory=list)

    def add(self, msg: str) -> None:
        self.items.append(msg)

    def extend(self, msgs) -> None:
        self.items.extend(msgs)


# ---------------------------------------------------------------------------
# artist inventory
# ---------------------------------------------------------------------------
def _is_reference_line(line) -> bool:
    """Constant axhlines/axvlines (2-point, constant y or x)."""
    x, y = line.get_xdata(orig=False), line.get_ydata(orig=False)
    if len(x) == 0:
        return True
    try:
        if len(x) == 2 and len(y) == 2 and (
                np.ptp(np.asarray(y, float)) == 0.0
                or np.ptp(np.asarray(x, float)) == 0.0):
            return True
    except TypeError:
        return False
    return False


def _finite(a) -> np.ndarray:
    a = np.asarray(a, dtype=float).ravel()
    return a[np.isfinite(a)]


def _eb_parts(cont):
    """(data_line, caplines, barlinecols) across matplotlib versions.

    mpl >= 3.10 exposes ``ErrorbarContainer.lines = (data_line, caplines,
    barlinecols)``; older versions expose ``data_line``/``caplines``/
    ``barlinecols`` attributes.
    """
    inner = getattr(cont, "lines", None)
    if inner is not None and len(inner) == 3:
        return inner[0], tuple(inner[1] or ()), tuple(inner[2] or ())
    return (getattr(cont, "data_line", None),
            tuple(getattr(cont, "caplines", []) or ()),
            tuple(getattr(cont, "barlinecols", []) or ()))


def collect_panel_artists(ax) -> dict:
    """Data-bearing artists of one axes, split by kind.

    Errorbar containers own their data line and cap lines; those inner
    Line2D artists are also registered in ``ax.lines`` (with a ``_nolegend_``-
    style label), so they must not be counted as separate series.
    """
    lines, bars, err = [], [], []
    refs = []
    inner = set()
    for cont in ax.containers:
        if type(cont).__name__ == "ErrorbarContainer":
            err.append(cont)
            dl, caps, _bcs = _eb_parts(cont)
            if dl is not None:
                inner.add(id(dl))
            for cap in caps:
                inner.add(id(cap))
    for line in ax.get_lines():
        label = line.get_label() or ""
        if id(line) in inner:
            continue
        if label.startswith("_"):
            continue
        if _is_reference_line(line):
            refs.append(line)
            continue
        lines.append(line)
    for cont in ax.containers:
        if type(cont).__name__ == "BarContainer":
            bars.append(cont)
    return {"lines": lines, "bars": bars, "errorbars": err, "refs": refs}


def _series_artists(ax) -> list:
    """One entry per plotted series (containers preferred over bare lines)."""
    art = collect_panel_artists(ax)
    if art["errorbars"]:
        return list(art["errorbars"])
    return list(art["lines"])


def _series_values(ax) -> dict:
    """Every plotted value per kind, as finite float arrays (no duplicates)."""
    art = collect_panel_artists(ax)
    out = {"line_values": [], "line_endpoints": [], "bar_values": []}
    for line in art["lines"]:
        v = _finite(line.get_ydata(orig=False))
        if len(v):
            out["line_values"].append(v)
    for cont in art["errorbars"]:
        # data_line carries the mean; caps and barlinecols carry (lo, hi)
        dl, caps, bcs = _eb_parts(cont)
        if dl is not None:
            v = _finite(dl.get_ydata(orig=False))
            if len(v):
                out["line_values"].append(v)
        for cap in caps:
            v = _finite(cap.get_ydata(orig=False))
            if len(v):
                out["line_endpoints"].append(v)
        for bc in bcs:
            for seg in bc.get_segments():
                seg = np.asarray(seg, float)
                if seg.size:
                    out["line_endpoints"].append(_finite(seg[:, 1]))
    for bc in art["bars"]:
        vals = np.asarray([b.get_height() for b in bc.patches], dtype=float)
        if bc.orientation == "horizontal":
            vals = np.asarray([b.get_width() for b in bc.patches], dtype=float)
        out["bar_values"].append(_finite(vals))
    return out


# ---------------------------------------------------------------------------
# contract 1: rendered-artist data contract
# ---------------------------------------------------------------------------
def _check_data_contract(ax, spec: dict, fails: Failures) -> None:
    metric = spec.get("metric")
    where = f"panel[{metric}]"
    art = collect_panel_artists(ax)
    n_lines = len(_series_artists(ax))
    n_bars = sum(len(bc.patches) for bc in art["bars"])

    exp_lines = spec.get("expect_lines")
    exp_bars = spec.get("expect_bars")
    if exp_lines is not None and n_lines != exp_lines:
        fails.add(f"{where}: {n_lines} plotted series, expected {exp_lines} "
                  f"(unfiltered overlay or missing series)")
    if exp_bars is not None and n_bars != exp_bars:
        fails.add(f"{where}: {n_bars} bars, expected {exp_bars}")

    pp = spec.get("points_per_line")
    if pp is not None:
        vals = _series_values(ax)
        for i, v in enumerate(vals["line_values"]):
            if len(v) != pp:
                fails.add(f"{where}: series {i} plots {len(v)} points, "
                          f"expected {pp}")

    side = spec.get("sidecar")
    if side is not None:
        sel = side[side["metric"] == metric] if "metric" in side.columns else side
        for col, val in (spec.get("filters") or {}).items():
            sel = sel[sel[col] == val]
        vcol = spec.get("value_col", "value")
        lo_col, hi_col = spec.get("ci_cols", ("ci_lo", "ci_hi"))
        if len(sel) == 0:
            fails.add(f"{where}: sidecar has zero rows for metric {metric!r}")
            return
        ref_v = _finite(sel[vcol].to_numpy(float))
        ref_lo = _finite(sel[lo_col].to_numpy(float)) if lo_col in sel else ref_v
        ref_hi = _finite(sel[hi_col].to_numpy(float)) if hi_col in sel else ref_v
        ref_min, ref_max = float(min(ref_lo.min(), ref_v.min())), \
            float(max(ref_hi.max(), ref_v.max()))
        vals = _series_values(ax)
        plotted = [v for v in vals["line_values"] + vals["bar_values"]
                   if len(v)]
        ends = [v for v in vals["line_endpoints"] if len(v)]
        if not plotted:
            fails.add(f"{where}: no plotted values found")
            return
        all_v = np.concatenate(plotted)
        got_min, got_max = float(all_v.min()), float(all_v.max())
        tol = spec.get("tol", 1e-9)
        if ends:
            all_e = np.concatenate(ends)
            got_min = min(got_min, float(all_e.min()))
            got_max = max(got_max, float(all_e.max()))
        if abs(got_min - ref_min) > tol:
            fails.add(f"{where}: plotted min {got_min:.6g} != sidecar min "
                      f"{ref_min:.6g} (tol {tol:g})")
        if abs(got_max - ref_max) > tol:
            fails.add(f"{where}: plotted max {got_max:.6g} != sidecar max "
                      f"{ref_max:.6g} (tol {tol:g})")
        # value-multiset check: every plotted mean must exist in the sidecar
        ref_all = np.sort(ref_v)
        for v in plotted:
            for x in _finite(v):
                if not np.any(np.abs(ref_all - x) <= max(tol, 1e-9)):
                    fails.add(f"{where}: plotted value {x:.6g} not in sidecar "
                              f"for this metric")
                    break

    # headroom
    if not spec.get("skip_headroom", False):
        lo, hi = ax.get_ylim()
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            fails.add(f"{where}: degenerate ylim {lo, hi}")
            return
        span = hi - lo
        vals = _series_values(ax)
        pts = [v for v in vals["line_values"] + vals["bar_values"] if len(v)]
        pts += [v for v in vals["line_endpoints"] if len(v)]
        if pts:
            all_v = np.concatenate(pts)
            margin = spec.get("headroom", DEFAULT_HEADROOM) * span
            below = all_v[all_v < lo + margin]
            above = all_v[all_v > hi - margin]
            if below.size:
                fails.add(f"{where}: {below.size} plotted value(s) within "
                          f"{DEFAULT_HEADROOM:.0%} of the bottom spine "
                          f"(min {below.min():.6g}, ylim {lo:.6g})")
            if above.size:
                fails.add(f"{where}: {above.size} plotted value(s) within "
                          f"{DEFAULT_HEADROOM:.0%} of the top spine "
                          f"(max {above.max():.6g}, ylim {hi:.6g})")


# ---------------------------------------------------------------------------
# contract 2: collision-free layout
# ---------------------------------------------------------------------------
def _text_artists(fig, ax=None) -> list:
    texts = []
    if ax is not None:
        for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if t.get_text().strip():
                texts.append(t)
        for t in ax.texts:
            if t.get_text().strip() and t.get_visible():
                texts.append(t)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            if t.get_text().strip() and t.get_visible():
                texts.append(t)
        leg = ax.get_legend()
        if leg is not None:
            texts.extend([t for t in leg.get_texts() if t.get_text().strip()])
    for t in fig.texts:
        if t.get_text().strip() and t.get_visible():
            texts.append(t)
    # NOTE: matplotlib registers the suptitle in fig.texts AND in
    # fig._suptitle; adding it twice produced a bogus self-overlap failure.
    if (fig._suptitle is not None and fig._suptitle.get_text().strip()
            and fig._suptitle not in fig.texts):
        texts.append(fig._suptitle)
    for leg in fig.legends:
        texts.extend([t for t in leg.get_texts() if t.get_text().strip()])
    return texts


def _bbox(t, renderer):
    try:
        bb = t.get_window_extent(renderer=renderer)
    except Exception:  # noqa: BLE001
        return None
    if bb.width <= 0 or bb.height <= 0:
        return None
    return bb


def _boxes_overlap(a, b, tol=PIXEL_TOL) -> bool:
    return not (a.x1 - tol <= b.x0 or b.x1 - tol <= a.x0
                or a.y1 - tol <= b.y0 or b.y1 - tol <= a.y0)


def _segments_from_artist(artist, ax=None):
    """DISPLAY-space segments for text-vs-data collision testing.

    Text bounding boxes live in display coordinates, so every data-space
    artist (Line2D, errorbar barlinecols, errorbar caps) must be transformed
    through ``ax.transData``; bar patches already report display extents.
    (v3 fix: the first version left lines and error bars in data units, so
    line-over-text collisions silently never fired.)
    """
    segs = []
    tr = ax.transData if ax is not None else None

    def _to_disp(pts):
        arr = np.asarray(pts, dtype=float)
        return tr.transform(arr) if tr is not None else arr

    try:
        if type(artist).__name__ == "Line2D":
            xy = _to_disp(artist.get_xydata())
            if len(xy) >= 2:
                segs.extend([(xy[i], xy[i + 1]) for i in range(len(xy) - 1)])
            elif len(xy) == 1:
                segs.append((xy[0], xy[0]))
        elif type(artist).__name__ == "BarContainer":
            for p in artist.patches:
                bb = p.get_window_extent()
                segs.append(((bb.x0, bb.y0), (bb.x1, bb.y1)))
        elif type(artist).__name__ == "ErrorbarContainer":
            _dl, caps, bcs = _eb_parts(artist)
            for bc in bcs:
                for s in bc.get_segments():
                    s = _to_disp(s)
                    if len(s) >= 2:
                        segs.append((s[0], s[-1]))
            for cap in caps:
                xy = _to_disp(cap.get_xydata())
                if len(xy) >= 2:
                    segs.append((xy[0], xy[1]))
                elif len(xy) == 1:
                    segs.append((xy[0], xy[0]))
    except Exception:  # noqa: BLE001
        pass
    return segs


def _seg_hits_box(seg, bb, n=40) -> bool:
    (x0, y0), (x1, y1) = seg
    for t in np.linspace(0.0, 1.0, n):
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        if bb.x0 < x < bb.x1 and bb.y0 < y < bb.y1:
            return True
    return False


def _check_collisions(fig, ax, spec: dict, fails: Failures) -> None:
    metric = spec.get("metric")
    where = f"panel[{metric}]"
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    texts = _text_artists(fig, ax)
    boxes = [(t, _bbox(t, renderer)) for t in texts]
    boxes = [(t, b) for t, b in boxes if b is not None]

    # canvas clipping
    fbb = fig.bbox
    for t, b in boxes:
        if (b.x0 < fbb.x0 - CANVAS_TOL or b.y0 < fbb.y0 - CANVAS_TOL
                or b.x1 > fbb.x1 + CANVAS_TOL or b.y1 > fbb.y1 + CANVAS_TOL):
            fails.add(f"{where}: text clipped by canvas: {t.get_text()[:40]!r}")

    # pairwise overlap (includes text-vs-title)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ti, bi = boxes[i]
            tj, bj = boxes[j]
            if _boxes_overlap(bi, bj):
                fails.add(f"{where}: text overlap: {ti.get_text()[:30]!r} x "
                          f"{tj.get_text()[:30]!r}")

    # text vs data geometry
    if ax is not None:
        artists = (collect_panel_artists(ax)["lines"]
                   + collect_panel_artists(ax)["bars"]
                   + collect_panel_artists(ax)["errorbars"]
                   + collect_panel_artists(ax)["refs"])
        segs = []
        for a in artists:
            segs.extend(_segments_from_artist(a, ax))
        leg = ax.get_legend()
        leg_ok = spec.get("legend_over_data_ok", False)
        for t, b in boxes:
            in_legend = (leg is not None and t in leg.get_texts())
            if in_legend and not leg_ok:
                inside = any(_seg_hits_box(s, b) for s in segs)
                if inside:
                    fails.add(f"{where}: legend text over data: "
                              f"{t.get_text()[:30]!r}")
                continue
            # tick labels live outside the axes; only check text inside the axes
            ab = ax.bbox
            if b.x1 < ab.x0 or b.x0 > ab.x1 or b.y1 < ab.y0 or b.y0 > ab.y1:
                continue
            if any(_seg_hits_box(s, b) for s in segs):
                fails.add(f"{where}: text over data: {t.get_text()[:40]!r}")

    # duplicate legend labels (per axes + per figure)
    legends = [ax.get_legend()] if ax is not None and ax.get_legend() else []
    legends += list(fig.legends)
    for leg in legends:
        labels = [t.get_text() for t in leg.get_texts()
                  if t.get_text().strip() and not t.get_text().startswith("_")]
        dupes = {x for x in labels if labels.count(x) > 1}
        if dupes:
            fails.add(f"{where}: duplicate legend label(s): {sorted(dupes)}")


# ---------------------------------------------------------------------------
# bounded layout relief (transparent, reported)
# ---------------------------------------------------------------------------
def _texts_blocked(fig, t, ax) -> bool:
    """True if `t` collides with data geometry or another text artist."""
    if text_hits_data(ax, t):
        return True
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = _bbox(t, renderer)
    if bb is None:
        return False
    for other in _text_artists(fig, ax):
        if other is t:
            continue
        ob = _bbox(other, renderer)
        if ob is not None and _boxes_overlap(bb, ob):
            return True
    return False


def _relieve_text_over_data(fig, ax, max_pt: int = 28, step: int = 2) -> list[str]:
    """Nudge colliding ax.texts vertically by the smallest clean offset.

    Keeps every label anchored to its object (displacement is a few points),
    never moves labels horizontally, and reports each adjustment. This is a
    layout-relief step, not a contract weakening: a label that cannot be
    relieved inside ``max_pt`` stays a failure.
    """
    from matplotlib.transforms import ScaledTranslation
    adjusted = []
    fig.canvas.draw()
    for t in list(ax.texts):
        if not t.get_text().strip() or not t.get_visible():
            continue
        if not text_hits_data(ax, t):
            continue
        from matplotlib.text import Annotation
        base = t.get_transform()
        is_ann = isinstance(t, Annotation)
        base_ann = getattr(t, "xyann", None) if is_ann else None
        placed = False
        # offsets ordered by displacement magnitude: pure vertical first (a bar
        # body or a cap line is cleared by lifting the label), then horizontal,
        # then diagonal, up to ``max_pt``.
        cands = []
        for mag in range(step, max_pt + 1, step):
            cands += [(0, mag), (0, -mag), (mag, 0), (-mag, 0),
                      (mag, mag), (-mag, mag), (mag, -mag), (-mag, -mag)]
        seen = set()
        for dx, dy in cands:
            if (dx, dy) in seen:
                continue
            seen.add((dx, dy))
            if is_ann and base_ann is not None:
                # Annotation text offset is in offset points: move xyann
                t.xyann = (base_ann[0] + dx, base_ann[1] + dy)
            else:
                t.set_transform(base + ScaledTranslation(
                    dx / 72.0, dy / 72.0, fig.dpi_scale_trans))
            fig.canvas.draw()
            if not _texts_blocked(fig, t, ax):
                adjusted.append(
                    f"{t.get_text()[:24]!r} moved ({dx:+d},{dy:+d}) pt")
                placed = True
                break
        if not placed:
            if is_ann and base_ann is not None:
                t.xyann = base_ann
            else:
                t.set_transform(base)
    return adjusted


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def assert_figure_clean(fig, panels, tol: float = 1e-9, strict: bool = True) -> dict:
    """Run both contracts; raise AssertionError on any failure (strict).

    Returns a report dict (per-panel failure lists + total).
    """
    fails = Failures()
    report = {}
    for spec in panels:
        ax = spec["ax"]
        local = Failures()
        _check_data_contract(ax, spec, local)
        _check_collisions(fig, ax, spec, local)
        if local.items:
            # bounded relief first: nudge colliding labels a few points and
            # re-test; remaining violations are still hard failures.
            moved = _relieve_text_over_data(fig, ax)
            if moved:
                retry = Failures()
                _check_data_contract(ax, spec, retry)
                _check_collisions(fig, ax, spec, retry)
                local = retry
                report.setdefault("_relieved", []).extend(moved)
        key = spec.get("metric", f"panel_{id(ax)}")
        report[key] = local.items
        fails.extend(local.items)
    report["_total"] = len(fails.items)
    if strict and fails.items:
        raise AssertionError(
            "visual_qa failures (%d):\n  - %s" % (
                len(fails.items), "\n  - ".join(fails.items[:25])))
    return report


def data_segments(ax) -> list:
    """All data-bearing segments of one axes, in DISPLAY coordinates."""
    art = collect_panel_artists(ax)
    segs = []
    for a in (art["lines"] + art["bars"] + art["errorbars"] + art["refs"]):
        segs.extend(_segments_from_artist(a, ax))
    return segs


def text_hits_data(ax, text) -> bool:
    """True if the text bbox intersects any plotted geometry of ``ax``.

    Call after ``fig.canvas.draw()`` (the bbox needs a renderer).
    """
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bb = _bbox(text, renderer)
    if bb is None:
        return False
    return any(_seg_hits_box(s, bb) for s in data_segments(ax))


def audit_figure_file(png_path, panel_specs=None) -> dict:  # pragma: no cover
    """Placeholder hook: audits are done on live figures, not on PNG pixels."""
    raise NotImplementedError("call assert_figure_clean on the live figure")
