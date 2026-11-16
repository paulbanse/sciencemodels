"""Standardized diagram templates (house style).

Purpose
-------
The three main chart types of applied work behind ONE house style, plus a
SERIES display that puts them side by side:

- ``styled_barchart``   -- categories vs. values, optional grouping
- ``styled_linechart``  -- a quantity along an axis, optional line groups
- ``styled_scatterplot`` -- two continuous variables, optional groups
- ``show_figures``      -- a series of figures SIDE BY SIDE in one row
  that never line-breaks (it scrolls horizontally) -- the diagram
  analogue of regression template RA. A cell is expected to produce a
  series of figures; this is how they show up next to each other.

INPUT  -- a tidy DataFrame plus a readable declaration of what is plotted:
          which column is which axis, and (optionally) which column groups.
OUTPUT -- one PNG per figure at the requested path, ``(fig, ax)`` for
          further tweaks, and the plotted numbers persisted as CSV
          (``write_figure_data``) -- a figure is only verifiable if its
          numbers survive as data.

Additional examples cover the recurring descriptive figure inventory behind
the same contracts: histogram with log axis, ECDF / log-log survival, Lorenz
curve, twin-axis decile chart, stacked composition, multi-panel composites;
the state choropleth lives in the sibling module ``diagram_maps.py``. The
three templates above remain the main archetypes.

The style constants are one dict: figsize around (8, 4.6), dpi 180, tight
layout, and the palette below, whose PRIME colors (purple, orange) lead the
group order. Table templates TA/TC and the house status prints are included
so the folder stays copy-pasteable on its own.

Dependencies: numpy, pandas, matplotlib. Optional: scipy (KDE overlay in
``styled_histogram``), geopandas (``diagram_maps``).
"""

from __future__ import annotations

import base64
import io
from html import escape
from pathlib import Path

import matplotlib

# No backend is forced here: headless script runs fall back to Agg on their
# own, and notebooks keep their inline backend so figures render in-cell.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


# --- BEGIN shared core (generated from _shared/template_core.py) ---
#: The baseline palette: six hues, each in a STRONG/NEON and a LIGHT
#: variant -- 12 baseline colors. The single source for every color the
#: templates use; reference it (PALETTE["blue"]) rather than hex literals.
PALETTE = {
    "purple": "#6A51A3", "purple_light": "#C9B8E8",
    "orange": "#F97316", "orange_light": "#FDBA74",
    "green": "#1A9850", "green_light": "#B7E4C7",
    "blue": "#2C7FB8", "blue_light": "#9ECAE1",
    "pink": "#EC4899", "pink_light": "#F9A8D4",
    "gray": "#636363", "gray_light": "#BDBDBD",
}

#: The status/annotation hue: house blue. Status symbols (``info``,
#: ``header``), figure captions and specification sentences are blue in
#: every module -- purple is the TABLE marker and the prime group color,
#: never the status color.
_INFO_COLOR = PALETTE["blue"]

#: Dark purple: the header marker of every standardized table (template TA).
_TABLE_MARKER = PALETTE["purple"]

#: Failure red lives deliberately OUTSIDE the palette (which has no red):
#: a failure must read as red.
_FAIL_COLOR = "#D73027"


def _p_text(p: float, digits: int = 4) -> str:
    """A p-value for DISPLAY: never collapses to "0" (a nonzero p is small).

    ``0.00004`` shows as ``"<0.0001"``; the stored number stays exact, so
    sorting/filtering on the numeric column is unaffected.
    """
    if p is None or pd.isna(p):
        return ""
    floor = 10.0 ** -digits
    if p < floor:
        return f"<{floor:.{digits}f}"
    return f"{p:.{digits}f}"


def format_table(df: pd.DataFrame, precision: int = 4) -> pd.DataFrame:
    """A display copy of ``df``: numbers formatted, values NEVER altered.

    - numeric columns whose values are all whole numbers -> strings with
      thousands separators (``3000.0`` -> ``"3,000"``);
    - other float columns -> at most ``precision`` decimals, trailing
      zeros and the trailing decimal point removed (``"0.30"`` ->
      ``"0.3"``); plain ``f``-formatting, so no scientific notation at
      ordinary magnitudes;
    - a NONZERO value too small for ``precision`` never prints as ``0``:
      it prints ``"<0.0001"`` (``">-0.0001"`` if negative), so a
      p-value of 0.00004 is visibly tiny rather than visibly zero;
    - NaN -> ``"-"``;
    - object/string (and bool) columns pass through UNCHANGED.

    Rounding is display-only: every shown value equals the computed one
    to ``precision`` decimals (exactly for whole numbers).
    """
    if precision < 0:
        raise ValueError(f"precision must be >= 0, got {precision}")
    floor = 10.0 ** -precision
    out = df.copy()
    for col in out.columns:
        s = out[col]
        is_numeric = (pd.api.types.is_float_dtype(s)
                      or pd.api.types.is_integer_dtype(s))
        if not is_numeric or pd.api.types.is_bool_dtype(s):
            continue  # object/string/bool columns pass through unchanged
        non_na = s.dropna()
        whole = len(non_na) > 0 and bool((non_na == np.floor(non_na)).all())

        def fmt(v):
            if pd.isna(v):
                return "-"
            if whole:
                return f"{int(v):,}"
            text = f"{float(v):.{precision}f}"
            if float(text) == 0.0 and v != 0:
                return (f">-{floor:.{precision}f}" if v < 0
                        else f"<{floor:.{precision}f}")
            text = text.rstrip("0").rstrip(".")
            return text if text not in ("", "-") else "0"

        out[col] = [fmt(v) for v in s]
    return out


def _in_notebook() -> bool:
    """True inside an IPython kernel (notebook/console), False in a script."""
    try:
        get_ipython  # noqa: F821 -- defined by IPython, not imported
    except NameError:
        return False
    return True


def _emit(symbol: str, color: str, msg: str, *, bold: bool = False) -> None:
    """One status line: colored HTML in a notebook, plain text in a script."""
    if not _in_notebook():
        print(f"{symbol} {msg}")
        return
    from IPython.display import HTML, display
    weight = "bold" if bold else "normal"
    display(HTML(
        f'<div style="font-family: monospace; font-weight: {weight}">'
        f'<span style="color: {color}">{symbol}</span> {escape(msg)}</div>'
    ))


def ok(msg: str) -> None:
    """A passing check."""
    _emit("\u2713", PALETTE["green"], msg)


def fail(msg: str) -> None:
    """A failing check."""
    _emit("\u2717", _FAIL_COLOR, msg)


def info(msg: str) -> None:
    """A neutral status line (paths, counts, progress)."""
    _emit("\u25b8", _INFO_COLOR, msg)


def header(msg: str) -> None:
    """A section header in the output stream."""
    _emit("\u25a0", _INFO_COLOR, msg, bold=True)


def style_table(df: pd.DataFrame):
    """TA: a standard (non-regression) table in the house style.

    Standard text/html with the dark-purple marker: the column-header
    line in dark purple (white bold text) -- the visual signature that
    this is a standardized output.
    """
    styled = df.style.set_table_styles([{
        "selector": "th",
        "props": [("background-color", _TABLE_MARKER),
                  ("color", "white"), ("font-weight", "bold")],
    }])
    if isinstance(df.index, pd.RangeIndex):
        styled = styled.hide(axis="index")
    return styled


def show_table(df: pd.DataFrame, alternative: str = "A") -> None:
    """Display a standard (non-regression) table: template TA or TC.

    - ``"A"`` (TA): house text/html with the dark-purple header marker --
      the default display for standard tables.
    - ``"C"`` (TC): the PLAIN DataFrame display -- the form VS Code's
      built-in Data Wrangler (and itables) hook into.

    ONE output; in a script, prints the plain table instead.
    """
    if alternative.upper() == "C":
        if not _in_notebook():
            print(df.to_string())
            return
        from IPython.display import display
        display(df)
        return
    styled = style_table(df)
    if not _in_notebook():
        print(df.to_string())
        return
    from IPython.display import HTML, display
    display(HTML(styled.to_html()))  # pure html: no text/plain repr


def table_html_wrapper(df: pd.DataFrame, *, title: str = "Table") -> str:
    """TC: a standard table as a COMPLETE standalone HTML document."""
    return style_table(df).to_html(title=title)


def write_table_html(
    df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "Table",
) -> Path:
    """TC-persist: write a standard table as a standalone HTML document
    (the dark-purple header marker included); return the path."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(table_html_wrapper(df, title=title), encoding="utf-8")
    return outpath


def show_palette() -> None:
    """The 12 baseline colors as a swatch row (strong | light per hue).

    A quick visual check of the house palette; ONE output.
    """
    if not _in_notebook():
        for name, hex_ in PALETTE.items():
            print(f"{name:14s} {hex_}")
        return
    from IPython.display import HTML, display
    hues = ["purple", "orange", "green", "blue", "pink", "gray"]
    panels = []
    for hue in hues:
        swatches = "".join(
            f'<div style="width: 56px; height: 28px; '
            f'background-color: {PALETTE[variant]}"></div>'
            for variant in (hue, f"{hue}_light")
        )
        panels.append(
            '<div style="flex: 0 0 auto; display: flex; flex-direction: '
            f'column; gap: 4px; font-family: monospace; font-size: 0.85em">'
            f'<div style="display: flex; gap: 4px">{swatches}</div>'
            f'<div>{hue}</div></div>'
        )
    display(HTML(
        '<div style="display: flex; gap: 20px; align-items: flex-start; '
        'flex-wrap: nowrap; overflow-x: auto; padding-bottom: 4px">'
        + "".join(panels) + "</div>"
    ))
# --- END shared core ---

#: The baseline palette: six hues, each in a STRONG/NEON and a LIGHT
#: variant -- 12 baseline colors. The single source for every color the
#: templates use; reference it (PALETTE["blue"]) rather than hex literals.


#: The house style. Change HERE once, not in every script.
#: ``colors`` orders DISTINCT STRONG hues for group marks -- PURPLE AND
#: ORANGE are the prime colors and lead the list; co-plotted categories
#: must differ in HUE, not only in lightness (strong/light pairs of one
#: hue are near-indistinguishable for overlapping marks and for
#: color-vision deficiency). The LIGHT palette variants are for
#: backgrounds, shading and secondary elements, not for group marks.
HOUSE_STYLE = {
    "colors": [PALETTE["purple"], PALETTE["orange"], PALETTE["green"],
               PALETTE["pink"], PALETTE["blue"], PALETTE["gray"]],
    "figsize": (8, 4.6),
    "series_figsize": (4.4, 3.3),  # one panel of a side-by-side series
    "dpi": 180,
    "title_size": 12,
    "label_size": 10,
    "tick_size": 9,
    "note_size": 8,
    "bar_width": 0.8,          # total width occupied by one category's bars
    "value_label": True,       # print the value on top of each bar
}

# ---------------------------------------------------------------------------
# Shared cosmetics (one place, all chart types)
# ---------------------------------------------------------------------------

def _cosmetics(ax, s: dict, title: str, xlabel: str, ylabel: str,
               *, grid_axis: str = "y") -> None:
    """Title, labels, spines, grid -- identical across chart types.

    ``grid_axis`` follows the READING direction of the values: "y" when the
    levels are on x (bars, lines), "x" when the values run horizontally
    (coefficient plots), so the grid never crosses the marks it should
    support.
    """
    if grid_axis not in ("x", "y", "both"):
        raise ValueError(
            f"grid_axis={grid_axis!r}: use 'x', 'y' or 'both'")
    ax.set_title(title, fontsize=s["title_size"])
    ax.set_xlabel(xlabel, fontsize=s["label_size"])
    ax.set_ylabel(ylabel, fontsize=s["label_size"])
    ax.tick_params(labelsize=s["tick_size"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis if grid_axis != "both" else "both",
            alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)


def _finish(
    fig,
    s: dict,
    source: str | None,
    outpath: str | Path | None,
    rect: tuple[float, float, float, float] | None = None,
    close: bool = False,
) -> None:
    """Source note, tight layout, optional PNG export, optional close.

    ``rect`` reserves part of the canvas from the layout (e.g. a right
    margin for a legend drawn outside the axes). ``close=True`` closes the
    figure after exporting it -- pass it when the caller does NOT keep the
    returned ``(fig, ax)`` (loops!), or the pyplot registry accumulates one
    open figure per call.
    """
    if source:
        fig.text(0.01, 0.01, source, fontsize=s["note_size"],
                 color=PALETTE["gray"])
    if rect is None:
        rect = (0, 0.03 if source else 0, 1, 1)
    fig.tight_layout(rect=rect)
    if outpath:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=s["dpi"])
    if close:
        plt.close(fig)


def _wide(data: pd.DataFrame, x: str, y: str, group: str | None,
          *, sort_x: bool = False, max_colors: int = 6) -> pd.DataFrame:
    """Index = x categories, columns = group levels -- VALIDATED.

    A chart is only as honest as its input frame, so the two silent-reshape
    traps are closed here rather than in each caller:

    - duplicate (x) rows (no group) would be drawn as repeated categories
      and duplicate (x, group) rows would be AVERAGED by ``pivot_table``:
      both raise ``ValueError`` naming the offending cells;
    - a missing (x, group) cell is reported via ``info()`` and stays a GAP
      (an absent bar / a broken line segment, never a zero);
    - more group levels than available strong hues raises instead of
      quietly recycling colors (two levels sharing a color is a false
      legend) -- pass ``style={"colors": [...]}`` for a longer list.

    ``sort_x=True`` orders the categories by value (numeric or datetime x):
    a line chart over years must place 1990 and 2020 at their real distance,
    not at equal spacing.
    """
    if x not in data.columns or y not in data.columns:
        raise ValueError(f"columns {x!r}/{y!r}: not present in the frame")
    if group and group not in data.columns:
        raise ValueError(f"column {group!r}: not present in the frame")
    if len(data) == 0:
        raise ValueError(f"columns {x!r}/{y!r}: empty frame")

    keys = [x, group] if group else [x]
    dup = data.duplicated(subset=keys, keep=False)
    if dup.any():
        cells = data.loc[dup, keys].drop_duplicates().head(4).to_numpy().tolist()
        raise ValueError(
            f"columns {keys}: {int(dup.sum())} duplicate rows -- one row per "
            f"{' x '.join(map(repr, keys))} expected, e.g. {cells}; "
            "aggregate (mean/sum) the frame first if that is intended")

    if group:
        wide = data.pivot_table(index=x, columns=group, values=y,
                                observed=True)
        # Legend/color order = APPEARANCE order, not pandas' alphabetical one
        wide = wide.reindex(columns=list(dict.fromkeys(data[group])))
        missing = int(wide.isna().to_numpy().sum())
        if missing:
            info(f"{y}: {missing} missing (x, group) cells left as gaps")
    else:
        wide = data.set_index(x)[[y]]

    order = list(dict.fromkeys(data[x]))
    if sort_x:
        order = sorted(order)
    wide = wide.reindex(order)

    if wide.shape[1] > max_colors:
        raise ValueError(
            f"column {group!r}: {wide.shape[1]} levels but only {max_colors} "
            "distinct strong hues -- raise the palette via "
            'style={"colors": [...]} or facet the groups into panels')
    return wide


# ---------------------------------------------------------------------------
# The three chart templates
# ---------------------------------------------------------------------------

def styled_barchart(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    source: str | None = None,
    value_fmt: str = "{:.2f}",
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    outpath: str | Path | None = None,
    close: bool = False,
    style: dict | None = None,
):
    """A simple or grouped bar chart in the house style.

    ``data`` is tidy: one row per bar (per x, or per x x group). ``group``
    splits each category into side-by-side bars, one color per level.
    Returns ``(fig, ax)``; nothing is shown or closed.
    """
    s = {**HOUSE_STYLE, **(style or {})}
    wide = _wide(data, x, y, group, max_colors=len(s["colors"]))

    n_cat, n_grp = wide.shape
    width = s["bar_width"] / n_grp
    centers = np.arange(n_cat)

    fig, ax = plt.subplots(figsize=figsize or s["figsize"])
    for j, level in enumerate(wide.columns):
        heights = wide[level].to_numpy(float)
        offset = (j - (n_grp - 1) / 2) * width
        bars = ax.bar(
            centers + offset, heights, width=width,
            color=s["colors"][j % len(s["colors"])],
            label=str(level) if group else None,
        )
        if s["value_label"]:
            ax.bar_label(bars, fmt=lambda v: value_fmt.format(v),
                         fontsize=s["note_size"], padding=2)

    ax.set_xticks(centers, [str(c) for c in wide.index], fontsize=s["tick_size"])
    _cosmetics(ax, s, title, xlabel, ylabel)
    if group:
        ax.legend(fontsize=s["tick_size"], frameon=False)
    if ylim:
        ax.set_ylim(*ylim)
    else:  # headroom for the value labels -- on the SIDE the data goes
        vals = wide.to_numpy(float)
        if not np.isfinite(vals).any():
            raise ValueError(f"column {y!r}: no finite values to plot")
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        lo, hi = min(0.0, lo) * 1.15, max(0.0, hi) * 1.15
        ax.set_ylim(lo, hi if hi > lo else lo + 1.0)
    _finish(fig, s, source, outpath, close=close)
    return fig, ax


def styled_linechart(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str | None = None,
    order: str = "auto",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    source: str | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    outpath: str | Path | None = None,
    close: bool = False,
    style: dict | None = None,
):
    """A line chart (with markers) in the house style.

    ``data`` is tidy: one row per point (per x, or per x x group).
    ``group`` draws one line per level. ``order`` controls the x axis:
    ``"auto"`` (default) sorts a NUMERIC or DATETIME x -- years land at
    their real distance, so 1990 and 2020 are not one unit apart -- and
    keeps the input order for an ordinary categorical x; ``"as-is"`` keeps
    the input order always; ``"sorted"`` always sorts. Returns ``(fig, ax)``.
    """
    s = {**HOUSE_STYLE, **(style or {})}
    if order not in ("auto", "as-is", "sorted"):
        raise ValueError(
            f"order={order!r}: use 'auto', 'as-is' or 'sorted'")
    xs_col = data[x]
    is_time = pd.api.types.is_datetime64_any_dtype(xs_col)
    sort_x = (order == "sorted"
              or (order == "auto"
                  and (pd.api.types.is_numeric_dtype(xs_col) or is_time)))
    wide = _wide(data, x, y, group, sort_x=sort_x,
                 max_colors=len(s["colors"]))

    # GEOMETRY: a sorted numeric/datetime x is plotted at its real values
    # (1990 and 2020 sit twenty years apart, not one unit); a categorical
    # x keeps an evenly spaced axis with the labels as ticks.
    numeric_axis = sort_x and (is_time
                               or pd.api.types.is_numeric_dtype(xs_col))
    if numeric_axis:
        xs = (wide.index.to_numpy() if is_time
              else wide.index.to_numpy(dtype=float))
    else:
        xs = np.arange(len(wide))

    fig, ax = plt.subplots(figsize=figsize or s["figsize"])
    for j, level in enumerate(wide.columns):
        ax.plot(
            xs, wide[level].to_numpy(float), marker="o",
            color=s["colors"][j % len(s["colors"])],
            label=str(level) if group else None,
            linewidth=1.8, markersize=4.5,
        )

    if not numeric_axis:
        ax.set_xticks(xs, [str(c) for c in wide.index],
                      fontsize=s["tick_size"])
    elif len(xs) <= 12:
        # A FEW numeric points (quintiles, years): tick AT the data. Letting
        # AutoLocator choose would put 0.5/1.5 half-steps on a quintile axis.
        labels = ([pd.Timestamp(v).strftime("%Y-%m-%d") for v in xs]
                  if is_time else [f"{float(v):g}" for v in xs])
        ax.set_xticks(xs, labels, fontsize=s["tick_size"])
    elif is_time:
        import matplotlib.dates as mdates
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.tick_params(labelsize=s["tick_size"])
    else:
        # many points: matplotlib's AutoLocator picks readable numeric ticks
        ax.tick_params(labelsize=s["tick_size"])
    _cosmetics(ax, s, title, xlabel, ylabel)
    if group:
        ax.legend(fontsize=s["tick_size"], frameon=False)
    if ylim:
        ax.set_ylim(*ylim)
    _finish(fig, s, source, outpath, close=close)
    return fig, ax


def styled_scatterplot(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    source: str | None = None,
    alpha: float = 0.45,
    size: float = 16,
    figsize: tuple[float, float] | None = None,
    outpath: str | Path | None = None,
    close: bool = False,
    style: dict | None = None,
):
    """A scatterplot in the house style.

    ``data`` is tidy: one row per point. ``group`` colors the points by
    level and adds a legend. Returns ``(fig, ax)``.
    """
    s = {**HOUSE_STYLE, **(style or {})}
    for col in (x, y):
        if col not in data.columns:
            raise ValueError(f"column {col!r}: not present in the frame")
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(
                f"column {col!r}: must be numeric, got dtype {data[col].dtype}")

    fig, ax = plt.subplots(figsize=figsize or s["figsize"])
    if group:
        # APPEARANCE order (a groupby would sort alphabetically and hand the
        # prime color to the wrong level); one hue per level, never recycled
        levels = list(dict.fromkeys(data[group].dropna()))
        if len(levels) > len(s["colors"]):
            raise ValueError(
                f"column {group!r}: {len(levels)} levels but only "
                f"{len(s['colors'])} distinct strong hues -- raise the "
                'palette via style={"colors": [...]} or facet into panels')
        for j, level in enumerate(levels):
            g = data[data[group] == level]
            ax.scatter(
                g[x], g[y], s=size, alpha=alpha, edgecolors="none",
                color=s["colors"][j], label=str(level),
            )
        # A framed, translucent legend: frameless text disappears into a
        # dense point cloud.
        ax.legend(fontsize=s["tick_size"], frameon=True, framealpha=0.85,
                  edgecolor="none", markerscale=2.0)
    else:
        ax.scatter(data[x], data[y], s=size, alpha=alpha,
                   color=s["colors"][0], edgecolors="none")

    _cosmetics(ax, s, title, xlabel, ylabel)
    _finish(fig, s, source, outpath, close=close)
    return fig, ax


# ---------------------------------------------------------------------------
# ADDITIONAL EXAMPLES: the econometric figure templates.
#
# Estimation lives in P_regression_template; these two functions DRAW its
# numbers. Both take a tidy frame of point estimates (one row per term, or
# one row per event-time bin), validate it at the door, and persist the
# plotted values next to the PNG. Same contracts as the archetypes above.
# ---------------------------------------------------------------------------

#: The treatment marker: deliberately OUTSIDE the palette. A vertical
#: "policy starts here" line must not read as a series colour.
_TREATMENT_COLOR = "#D73027"

#: Two-sided normal critical values for the confidence levels applied work
#: actually uses. 0.95 -> 1.96 exactly (house convention); any other level
#: goes through scipy when it is importable, else ``crit=`` must be given.
_CRIT = {0.90: 1.6449, 0.95: 1.96, 0.99: 2.5758}


def _crit_for(level: float, crit: float | None) -> float:
    """The multiplier applied to a standard error (1.96 at 95%)."""
    if crit is not None:
        crit = float(crit)
        if not np.isfinite(crit) or crit <= 0:
            raise ValueError(f"crit={crit!r}: must be a positive number")
        return crit
    if not 0 < level < 1:
        raise ValueError(f"level={level!r}: must be a confidence level "
                         "between 0 and 1 (0.95, not 95)")
    if level in _CRIT:
        return _CRIT[level]
    try:
        from scipy.stats import norm
    except ImportError:  # scipy is optional in this module
        raise ValueError(
            f"level={level!r}: only {sorted(_CRIT)} are tabulated -- install "
            "scipy or pass crit=... explicitly")
    return float(np.round(norm.ppf(0.5 + level / 2), 4))


def _validate_points(data: pd.DataFrame, keys: list[str], cols: list[str],
                     what: str) -> None:
    """Shared door check for the point-estimate charts.

    One row per key combination: two estimates for the same term would be
    drawn on top of each other (a chart nobody can read, silently), so the
    duplicate case RAISES instead of averaging as a pivot would.
    """
    if len(data) == 0:
        raise ValueError(f"{what}: empty frame")
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"{what}: columns {missing} not present in the frame")
    for col in cols:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(
                f"{what}: column {col!r} must be numeric, got dtype "
                f"{data[col].dtype}")
    dup = data.duplicated(subset=keys, keep=False)
    if dup.any():
        cells = data.loc[dup, keys].drop_duplicates().head(4).to_numpy().tolist()
        raise ValueError(
            f"{what}: {int(dup.sum())} duplicate rows for {keys} -- one row "
            f"per {' x '.join(map(repr, keys))} expected, e.g. {cells}; "
            "pick one specification instead of plotting both")


def _stars(p: float) -> str:
    """Conventional significance marks (10%, 5%, 1%)."""
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def styled_coefplot(
    data: pd.DataFrame,
    *,
    y: str = "term",
    x: str = "coef",
    xerr: str = "se",
    group: str | None = None,
    p: str | None = None,
    level: float = 0.95,
    crit: float | None = None,
    reference: float = 0.0,
    annotate: bool = True,
    value_fmt: str = "{:+.3f}",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    source: str | None = None,
    xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    outpath: str | Path | None = None,
    close: bool = False,
    style: dict | None = None,
):
    """Point estimates with confidence intervals -- a coefficient plot.

    ``data`` is tidy: one row per term (per term x group when ``group`` is
    given). Estimates sit on the x axis and terms on the y axis, the FIRST
    row at the TOP, so the picture reads like a regression table.

    - the interval is ``estimate +/- crit * se`` (``crit = 1.96`` at
      ``level=0.95``); a row whose ``xerr`` is missing is drawn as a point
      WITHOUT an interval and reported, never as a zero-width interval;
    - a dashed line at ``reference`` (0 = no effect) shows what "nothing"
      means: a term whose interval crosses it is not distinguishable from it;
    - ``group`` dodges the terms vertically, one strong hue per level in
      appearance order (the prime colors purple, orange, green, pink) -- two
      specifications of the same term side by side;
    - with ``p`` the estimate is annotated with ``*/**/***`` next to its
      value; ``annotate=False`` drops the numbers.

    Returns ``(fig, ax)``; with ``outpath`` the PNG and its ``_data.csv``
    (estimate, se, lo, hi, position) are written.
    """
    s = {**HOUSE_STYLE, **(style or {})}
    keys = [y] + ([group] if group else [])
    required = [x, xerr] + ([p] if p else [])
    _validate_points(data, keys, required, what="coefplot")
    z = _crit_for(level, crit)

    df = data.copy()
    terms = list(dict.fromkeys(df[y]))
    levels = list(dict.fromkeys(df[group])) if group else [None]
    if group and len(levels) > len(s["colors"]):
        raise ValueError(
            f"column {group!r}: {len(levels)} levels but only "
            f"{len(s['colors'])} distinct strong hues -- raise the palette "
            'via style={"colors": [...]} or facet into panels')

    # Positions: first term at the TOP (top = n-1, axis ascending).
    span = 0.20  # vertical dodge per group level, in row units
    base = {term: len(terms) - 1 - i for i, term in enumerate(terms)}
    pos = df[y].map(base).to_numpy(float)
    if group:
        offset = {lv: (j - (len(levels) - 1) / 2) * span
                  for j, lv in enumerate(levels)}
        pos = pos + df[group].map(offset).to_numpy(float)
    df["_pos"] = pos
    df["lo"] = df[x] - z * df[xerr]
    df["hi"] = df[x] + z * df[xerr]

    fig, ax = plt.subplots(figsize=figsize or s["figsize"])
    ax.axvline(reference, color=PALETTE["gray"], linestyle="--", linewidth=1.0,
               zorder=1)

    handles = []
    for j, lv in enumerate(levels):
        sub = df[df[group] == lv] if group else df
        color = s["colors"][j]
        label = None if lv is None else str(lv)
        with_ci = sub[np.isfinite(sub[xerr])]
        if len(with_ci):
            handles.append(ax.errorbar(
                with_ci[x], with_ci["_pos"], xerr=z * with_ci[xerr], fmt="o",
                color=color, ecolor=color, capsize=3, elinewidth=1.2,
                markersize=5, label=label, zorder=3))
            label = None
        without = sub[~np.isfinite(sub[xerr])]
        if len(without):
            info(f"coefplot: {len(without)} row(s) without a finite {xerr!r} "
                 "drawn as points, no interval")
            handles.append(ax.plot(without[x], without["_pos"], "o",
                                   color=color, markerfacecolor="white",
                                   markersize=5, label=label, zorder=3)[0])

    ax.set_yticks([base[t] for t in terms], [str(t) for t in terms],
                  fontsize=s["tick_size"])
    ax.set_ylim(-0.6, len(terms) - 0.4)
    _cosmetics(ax, s, title, xlabel, ylabel, grid_axis="x")

    # Room for the annotations: they sit to the RIGHT of each interval, so
    # the x axis has to reserve space beyond the data.
    flat = np.concatenate([df["lo"].to_numpy(float), df["hi"].to_numpy(float),
                           df[x].to_numpy(float), np.array([reference])])
    flat = flat[np.isfinite(flat)]
    if not len(flat):
        raise ValueError(f"coefplot: no finite {x!r} values to plot")
    lo_v, hi_v = float(flat.min()), float(flat.max())
    width = (hi_v - lo_v) or 1.0
    pad = 0.12 * width
    left = min(lo_v, reference) - pad
    right = max(hi_v, reference) + pad + (0.34 * width if annotate else 0.0)
    if annotate:
        for _, row in df.iterrows():
            if not np.isfinite(row[x]):
                continue
            anchor = row["hi"] if np.isfinite(row["hi"]) else row[x]
            marks = _stars(row[p]) if p else ""
            ax.text(anchor + 0.03 * width, row["_pos"],
                    f"{value_fmt.format(row[x])}{marks}", va="center",
                    ha="left", fontsize=s["note_size"],
                    color=PALETTE["gray"])
    ax.set_xlim(*(xlim if xlim else (left, right)))
    if group:
        ax.legend(handles=handles, fontsize=s["tick_size"], frameon=False,
                  loc="best")

    if outpath is not None:
        export = pd.DataFrame({
            y: df[y].to_numpy(),
            **({group: df[group].to_numpy()} if group else {}),
            "estimate": df[x].to_numpy(), "se": df[xerr].to_numpy(),
            "level": level, "crit": z,
            "lo": df["lo"].to_numpy(), "hi": df["hi"].to_numpy(),
            "position": df["_pos"].to_numpy(),
        })
        if p:
            export["p"] = df[p].to_numpy()
            export["stars"] = [_stars(v) for v in df[p]]
        write_figure_data(export, outpath)

    _finish(fig, s, source, outpath, close=close)
    return fig, ax


def styled_eventstudy(
    data: pd.DataFrame,
    *,
    x: str = "e",
    y: str = "coef",
    yerr: str = "se",
    ref_e: float | None = -1,
    level: float = 0.95,
    crit: float | None = None,
    band: bool = True,
    label: str | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    source: str | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    outpath: str | Path | None = None,
    close: bool = False,
    style: dict | None = None,
):
    """A dynamic trajectory across event time -- an event-study plot.

    ``data`` is tidy: one row per event-time bin. Connected markers show the
    path and the shaded band (``alpha=0.15``) its interval; ``band=False``
    switches to vertical error bars. A dashed horizontal line marks zero (no
    effect) and a vertical line at ``x = 0`` marks the treatment.

    The OMITTED REFERENCE BIN is handled explicitly, because an event study
    always has one:

    - ``ref_e`` names it (-1 by convention: the period just before
      treatment). If the frame has no row for it, one is INSERTED with
      coefficient 0 and se 0 -- that is what "normalised to zero" means --
      and reported via ``info()``;
    - if the frame DOES carry a row for it, the values are kept as given; an
      estimated reference (se > 0 and a non-zero coefficient) is reported,
      since a visible interval at the omitted bin usually means the model did
      not normalise it;
    - its zero-width interval pinches the band at the reference: that pinch
      is the normalisation, not evidence of precision.

    Returns ``(fig, ax)``; with ``outpath`` the PNG and its ``_data.csv``
    (bin, estimate, se, lo, hi, is_reference) are written.
    """
    s = {**HOUSE_STYLE, **(style or {})}
    _validate_points(data, [x], [x, y, yerr], what="event study")
    z = _crit_for(level, crit)

    df = data.copy()
    finite_y = np.isfinite(df[y].to_numpy(float))
    if not finite_y.all():
        info(f"event study: {int((~finite_y).sum())} row(s) without a finite "
             f"{y!r} dropped")
        df = df[finite_y].copy()
        if len(df) == 0:
            raise ValueError(f"event study: no finite {y!r} values to plot")
    df["_ref"] = False

    if ref_e is not None:
        ref_e = float(ref_e)
        hit = np.isclose(df[x].to_numpy(float), ref_e)
        if hit.any():
            df.loc[hit, "_ref"] = True
            row = df.loc[hit].iloc[0]
            estimated = (np.isfinite(row[yerr])
                         and not np.isclose(row[yerr], 0.0)
                         and not np.isclose(row[y], 0.0))
            if estimated:
                info(f"event study: reference bin {ref_e:g} carries "
                     f"coef={row[y]:+.3f}, se={row[yerr]:.3f} -- kept as "
                     "given; an ESTIMATED reference usually means the model "
                     "did not normalise it")
        else:
            df = pd.concat(
                [df, pd.DataFrame({x: [ref_e], y: [0.0], yerr: [0.0],
                                   "_ref": [True]})], ignore_index=True)
            info(f"event study: reference bin {ref_e:g} was absent and is "
                 "inserted at 0 with no interval (normalised by "
                 "construction)")

    df = df.sort_values(x, kind="mergesort").reset_index(drop=True)
    xs = df[x].to_numpy(float)
    ys = df[y].to_numpy(float)
    lo = ys - z * df[yerr].to_numpy(float)
    hi = ys + z * df[yerr].to_numpy(float)
    color = s["colors"][0]  # the prime hue carries the trajectory

    fig, ax = plt.subplots(figsize=figsize or s["figsize"])
    ax.axhline(0.0, color=PALETTE["gray"], linestyle="--", linewidth=1.0,
               zorder=1)
    if band:
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0,
                        zorder=2)
    else:
        ax.errorbar(xs, ys, yerr=np.vstack([ys - lo, hi - ys]), fmt="none",
                    ecolor=color, elinewidth=1.1, capsize=3, alpha=0.8,
                    zorder=2)
    band_label = f"estimate ({level:.0%} CI)" if band else "estimate"
    ax.plot(xs, ys, marker="o", color=color, linewidth=1.4, markersize=5,
            label=band_label, zorder=3)
    # The omitted bin: hollow square, so the normalisation is visible; the
    # legend spells out what that marker means (otherwise a lone square at
    # zero reads as a data point with no interval).
    ref_rows = df["_ref"].to_numpy(bool)
    if ref_rows.any():
        ax.plot(xs[ref_rows], ys[ref_rows], marker="s", linestyle="none",
                markersize=6, markerfacecolor="white",
                markeredgecolor=PALETTE["gray"],
                label=f"reference bin (normalised to 0, e = {ref_e:g})",
                zorder=4)

    ax.axvline(0.0, color=_TREATMENT_COLOR, linestyle="--", linewidth=1.1,
               alpha=0.9, zorder=1)
    if label:
        # Offset a few points RIGHT of the line: text printed on top of a
        # dashed marker is unreadable at print size.
        ax.annotate(label, xy=(0.0, 0.985), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", rotation=90,
                    va="top", ha="left", fontsize=s["note_size"],
                    color=_TREATMENT_COLOR)

    if len(xs) <= 12:
        # A handful of bins (the normal case): tick AT the bins. Letting
        # matplotlib choose would put half-steps between event times.
        ax.set_xticks(xs, [f"{v:g}" for v in xs], fontsize=s["tick_size"])
    _cosmetics(ax, s, title, xlabel, ylabel)

    flat = np.concatenate([lo, hi, ys, np.array([0.0])])
    flat = flat[np.isfinite(flat)]
    pad = 0.12 * ((flat.max() - flat.min()) or 1.0)
    ax.set_ylim(*(ylim if ylim else (flat.min() - pad, flat.max() + pad)))
    if ref_rows.any():
        ax.legend(fontsize=s["tick_size"], frameon=False, loc="best")

    if outpath is not None:
        export = pd.DataFrame({
            "bin": xs, "estimate": ys, "se": df[yerr].to_numpy(),
            "level": level, "crit": z, "lo": lo, "hi": hi,
            "is_reference": ref_rows,
        })
        write_figure_data(export, outpath)

    _finish(fig, s, source, outpath, close=close)
    return fig, ax


# ---------------------------------------------------------------------------
# ADDITIONAL EXAMPLES: the recurring descriptive figures.
#
# The three templates above are the MAIN ARCHETYPES; the functions below
# are additional examples covering the recurring descriptive figures of
# applied descriptive work keeps asking for: histogram with log axis,
# ECDF and log-log survival, Lorenz curve, twin-axis decile chart, stacked
# composition and multi-panel composites. Same contracts
# as the archetypes: tidy DataFrame + keyword-only declaration of what
# is plotted, ``(fig, ax)`` back, PNG via ``outpath=``, nothing shown or
# closed, colors from ``PALETTE`` via ``s["colors"]``. The plotted
# numbers come back as tidy frames (``*_points``) for
# ``write_figure_data`` -- a figure is only verifiable if its numbers
# survive as data.
# ---------------------------------------------------------------------------

def _finite_pairs(
    data: pd.DataFrame, x: str, weights: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite x (and weight) rows as arrays; unit weights when unweighted.

    Missing and non-finite values -- and non-positive weights -- are
    dropped per series with a count reported via ``info()``; a series
    with no usable value raises ``ValueError`` naming the column.
    """
    xs = data[x].to_numpy(float)
    if weights:
        ws = data[weights].to_numpy(float)
        keep = np.isfinite(xs) & np.isfinite(ws) & (ws > 0)
    else:
        ws = np.ones(len(xs))
        keep = np.isfinite(xs)
    dropped = int((~keep).sum())
    if dropped:
        what = f" or non-positive weights in {weights!r}" if weights else ""
        info(f"{x}: dropped {dropped} rows with missing/non-finite values{what}")
    if not keep.any():
        raise ValueError(f"column {x!r}: no usable values (all missing)")
    return xs[keep], ws[keep]


def _kde_curve(
    xs: np.ndarray, w: np.ndarray, log_x: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian KDE of the plotted values on a 300-point grid.

    On a log axis the KDE runs in log10 space:
    the density on the original scale is the log-density divided by the
    Jacobian ``x * ln(10)``. Weights enter as SHARES (sum to 1).
    """
    # scipy is optional in the core module: only kde=True needs it.
    from scipy.stats import gaussian_kde

    if len(np.unique(xs)) < 2:
        raise ValueError("kde=True needs at least 2 distinct values")
    w = w / w.sum()
    if log_x:
        grid = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 300)
        pdf = gaussian_kde(np.log10(xs), weights=w)(np.log10(grid))
        return grid, pdf / (grid * np.log(10))
    grid = np.linspace(xs.min(), xs.max(), 300)
    return grid, gaussian_kde(xs, weights=w)(grid)


def histogram_points(
    data: pd.DataFrame,
    *,
    x: str,
    weights: str | None = None,
    bins: int = 50,
    log_x: bool = False,
    kde: bool = False,
    vlines: list[tuple[str, float]] | None = None,
    density: bool = True,
) -> pd.DataFrame:
    """The plotted numbers of ``styled_histogram`` as one tidy long frame.

    Rows carry a ``series`` tag: ``"hist"`` (``x`` = bin left edge,
    ``x2`` = bin right edge, ``y`` = bar height), ``"kde"`` (``x``,
    ``y`` = density at x) and ``"vline"`` (``x`` = value, ``label`` =
    annotation). Weights are NORMALIZED TO MEAN 1, so weighted counts
    stay on the sample-count scale and a density histogram integrates
    to 1. ``log_x`` drops non-positive values (count reported via
    ``info()``) and bins log-spaced.
    """
    xs, ws = _finite_pairs(data, x, weights)
    if log_x:
        pos = xs > 0
        dropped = int((~pos).sum())
        if dropped:
            info(f"{x}: dropped {dropped} non-positive values (log axis)")
        xs, ws = xs[pos], ws[pos]
        if len(xs) == 0:
            raise ValueError(f"column {x!r}: no positive values for a log axis")
        if not xs.min() < xs.max():
            raise ValueError(
                f"column {x!r}: kde/bins need 2 distinct positive values")
        edges = np.logspace(np.log10(xs.min()), np.log10(xs.max()), bins + 1)
    else:
        edges = np.histogram_bin_edges(xs, bins=bins)

    heights, edges = np.histogram(xs, bins=edges,
                                  weights=ws / ws.mean(), density=density)
    rows = [{"series": "hist", "x": edges[i], "x2": edges[i + 1],
             "y": heights[i], "label": ""} for i in range(len(heights))]

    if kde:
        kx, ky = _kde_curve(xs, ws, log_x)
        rows += [{"series": "kde", "x": a, "x2": np.nan, "y": b, "label": ""}
                 for a, b in zip(kx, ky)]

    for label, value in (vlines or []):
        rows.append({"series": "vline", "x": float(value), "x2": np.nan,
                     "y": np.nan, "label": label})
    return pd.DataFrame(rows, columns=["series", "x", "x2", "y", "label"])


def _draw_vlines(ax, s: dict, vlines: list[tuple[str, float]] | None) -> None:
    """Dashed markers annotated in the house gray (histogram/ECDF guides)."""
    for label, value in (vlines or []):
        ax.axvline(value, color=PALETTE["gray"], linestyle="--", linewidth=1.2)
        # Annotated in the house gray, pinned to the line's x position.
        ax.text(value, 0.98, label, rotation=90, va="top", ha="right",
                fontsize=s["note_size"], color=PALETTE["gray"],
                transform=ax.get_xaxis_transform())


def styled_histogram(
    data: pd.DataFrame,
    *,
    x: str,
    weights: str | None = None,
    bins: int = 50,
    log_x: bool = False,
    kde: bool = False,
    vlines: list[tuple[str, float]] | None = None,
    density: bool = True,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    source: str | None = None,
    figsize: tuple[float, float] | None = None,
    outpath: str | Path | None = None,
    close: bool = False,
    style: dict | None = None,
):
    """A weighted histogram in the house style.

    ``log_x`` sets a log x-axis with ABSOLUTE tick labels (1K, 10K, 1M;
    non-positive values dropped, count reported via ``info()``).
    ``kde=True`` overlays a Gaussian KDE (scipy) of the plotted values.
    ``vlines=[(label, value), ...]`` draws dashed markers annotated in
    the house gray (e.g. the median). Weights are normalized to mean 1
    (see ``histogram_points``). Returns ``(fig, ax)``.
    """
    s = {**HOUSE_STYLE, **(style or {})}
    pts = histogram_points(data, x=x, weights=weights, bins=bins,
                           log_x=log_x, kde=kde, vlines=vlines,
                           density=density)
    hist = pts[pts["series"] == "hist"]

    fig, ax = plt.subplots(figsize=figsize or s["figsize"])
    ax.bar(hist["x"], hist["y"], width=hist["x2"] - hist["x"],
           align="edge", color=s["colors"][0], edgecolor="white")
    if kde:
        curve = pts[pts["series"] == "kde"]
        ax.plot(curve["x"], curve["y"], color=s["colors"][1],
                linewidth=2.0, label="Kernel density")
        ax.legend(fontsize=s["tick_size"], frameon=False)
    _draw_vlines(ax, s, vlines)
    if log_x:
        ax.set_xscale("log")
        # Absolute (never relative) tick labels: 1K, 10K, 1M, ...
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _p: f"{v / 1e6:g}M" if v >= 1e6
            else f"{v / 1e3:g}K" if v >= 1e3 else f"{v:g}"))
    _cosmetics(ax, s, title, xlabel, ylabel)
    _finish(fig, s, source, outpath, close=close)
    return fig, ax


def ecdf_points(
    data: pd.DataFrame,
    *,
    x: str,
    weights: str | None = None,
    log_x: bool = False,
    survival: bool = False,
    loglog: bool = False,
) -> pd.DataFrame:
    """The plotted points of ``styled_ecdf`` as a tidy frame (x, share).

    The weighted empirical CDF at every sorted observation: ``share`` is
    the cumulative weight share (weights normalized to sum 1); with
    ``survival=True`` it is ``1 - F``. ``loglog=True`` (the Pareto tail
    view) drops non-positive points on either axis, ``log_x`` drops
    non-positive values -- each with a count reported via ``info()``.
    """
    xs, ws = _finite_pairs(data, x, weights)
    if log_x or loglog:
        pos = xs > 0
        dropped = int((~pos).sum())
        if dropped:
            info(f"{x}: dropped {dropped} non-positive values (log axes)")
        xs, ws = xs[pos], ws[pos]
        if len(xs) == 0:
            raise ValueError(
                f"column {x!r}: no positive values for log axes")
    order = np.argsort(xs, kind="mergesort")
    xs, ws = xs[order], ws[order]
    if survival:
        # P(X > x_i) as a REVERSE weight sum: 1 - F cancels
        # catastrophically at the top and would leave the upper
        # tail at 1e-14 instead of 1e-4.
        share = (np.cumsum(ws[::-1])[::-1] - ws) / ws.sum()
    else:
        share = np.cumsum(ws) / ws.sum()
    if loglog:
        pos = share > 0
        dropped = int((~pos).sum())
        if dropped:
            info(f"{x}: dropped {dropped} points with share <= 0 (log-log)")
        xs, share = xs[pos], share[pos]
        if len(xs) == 0:
            raise ValueError(
                f"column {x!r}: no positive shares for log-log axes")
    return pd.DataFrame({"x": xs, "share": share})


def pareto_tail(
    values, weights=None, tail_pct: float = 5.0,
) -> pd.DataFrame:
    """The Pareto tail of the log-log survival view, as one tidy row.

    Two numbers, clearly separated:

    - ``alpha_hill`` -- the weighted Hill MLE of the tail exponent over the
      top ``tail_pct`` percent: ``1 / mean_w(log(x / x_min))``. This is the
      quotable estimate and the one the figure annotates.
    - ``alpha_ols`` -- the log-log CCDF regression of the same tail
      (``log10 share above`` on ``log10 value``), reported for the VISUAL
      check only: the fitted line is what the eye reads, not an estimator
      (a regression of log x on log share has slope ``-1/alpha``, so the
      exponent is ``-1/slope`` -- reporting the raw slope as "alpha"
      inverts it, e.g. 0.56 where the true exponent is 1.6).

    The row carries ``tail_pct, n_tail, x_min, alpha_hill, alpha_ols,
    slope, intercept, r_squared`` for ``write_figure_data``.
    """
    x = np.asarray(values, dtype=float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0) & (x > 0)
    x, w = x[ok], w[ok]
    if len(x) < 2:
        raise ValueError("pareto_tail: need at least 2 positive values")
    order = np.argsort(x, kind="mergesort")
    x, w = x[order], w[order]
    # Survival as a reverse weight sum (exact at the top; see ecdf_points).
    share = (np.cumsum(w[::-1])[::-1] - w) / w.sum()
    tail = (share > 0) & (share <= tail_pct / 100)
    if tail.sum() < 2:
        raise ValueError(
            f"pareto_tail: fewer than 2 points in the top {tail_pct:g}%")
    x_tail, w_tail, share_tail = x[tail], w[tail], share[tail]
    x_min = float(x_tail.min())

    # Weighted Hill (MLE) tail exponent.
    alpha_hill = float(w_tail.sum()
                       / np.sum(w_tail * np.log(x_tail / x_min)))

    from scipy.stats import linregress  # optional dependency: kde/tail fits

    slope, intercept, r, _, _ = linregress(
        np.log10(share_tail), np.log10(x_tail))
    alpha_ols = float(-1.0 / slope) if slope != 0 else float("nan")
    return pd.DataFrame([{
        "tail_pct": float(tail_pct), "n_tail": int(tail.sum()),
        "x_min": x_min, "alpha_hill": alpha_hill, "alpha_ols": alpha_ols,
        "slope": float(slope), "intercept": float(intercept),
        "r_squared": float(r ** 2),
    }])


def styled_ecdf(
    data: pd.DataFrame,
    *,
    x: str,
    weights: str | None = None,
    log_x: bool = False,
    survival: bool = False,
    loglog: bool = False,
    vlines: list[tuple[str, float]] | None = None,
    tail_pct: float | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    source: str | None = None,
    figsize: tuple[float, float] | None = None,
    outpath: str | Path | None = None,
    close: bool = False,
    style: dict | None = None,
):
    """A weighted empirical CDF in the house style.

    ``survival=True`` plots ``1 - F`` instead of ``F``; ``loglog=True``
    puts both axes on log scales -- the Pareto tail view. ``vlines``
    draws dashed gray annotated markers (e.g. the median).
    ``tail_pct=5`` shades the top tail, overlays the log-log fitted line
    and annotates the Hill tail exponent (needs ``loglog=True``; see
    ``pareto_tail`` -- the line is a visual check, the Hill MLE is the
    quotable number). Returns ``(fig, ax)``.
    """
    s = {**HOUSE_STYLE, **(style or {})}
    if tail_pct is not None and not loglog:
        raise ValueError("tail_pct: needs loglog=True (the Pareto tail view)")
    pts = ecdf_points(data, x=x, weights=weights, log_x=log_x,
                      survival=survival, loglog=loglog)
    fig, ax = plt.subplots(figsize=figsize or s["figsize"])
    ax.step(pts["x"], pts["share"], where="post", color=s["colors"][0],
            linewidth=1.6)
    _draw_vlines(ax, s, vlines)
    if tail_pct is not None:
        # The tail region, the fitted line and the Hill tail exponent.
        fit = pareto_tail(
            data[x].to_numpy(float),
            data[weights].to_numpy(float) if weights else None,
            tail_pct).iloc[0]
        tail = pts[pts["share"] <= tail_pct / 100]
        ax.axvspan(tail["x"].min(), tail["x"].max(), alpha=0.15,
                   color=s["colors"][1],
                   label=f"Pareto tail (top {tail_pct:g}%)")
        xs = np.array([tail["x"].min(), tail["x"].max()])
        shares = 10 ** ((np.log10(xs) - fit["intercept"]) / fit["slope"])
        ax.plot(xs, shares, "--", color=s["colors"][1], linewidth=1.2,
                alpha=0.8, label="log-log fit (visual check)")
        ax.text(0.02, 0.06,
                f"Hill alpha (top {tail_pct:g}%) = {fit['alpha_hill']:.2f}",
                transform=ax.transAxes, fontsize=s["label_size"],
                color=s["colors"][1])
        ax.legend(fontsize=s["tick_size"], frameon=False)
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    elif log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _p: f"{v / 1e6:g}M" if v >= 1e6
            else f"{v / 1e3:g}K" if v >= 1e3 else f"{v:g}"))
    _cosmetics(ax, s, title, xlabel, ylabel)
    _finish(fig, s, source, outpath, close=close)
    return fig, ax


def lorenz_points(data, value, weights=None) -> pd.DataFrame:
    """Tidy Lorenz curve points: (population_share, value_share).

    One row per sorted observation plus the ``(0, 0)`` origin. Values
    are restricted to non-negative (negatives dropped, count reported
    via ``info()``); weights enter as population shares (sum to 1).
    """
    xs, ws = _finite_pairs(data, value, weights)
    nonneg = xs >= 0
    dropped = int((~nonneg).sum())
    if dropped:
        info(f"{value}: dropped {dropped} negative values (Lorenz undefined)")
    xs, ws = xs[nonneg], ws[nonneg]
    if len(xs) == 0:
        raise ValueError(
            f"column {value!r}: no non-negative values for a Lorenz curve")
    order = np.argsort(xs, kind="mergesort")
    xs, ws = xs[order], ws[order]
    mass = float((ws * xs).sum())
    if mass <= 0:
        raise ValueError(
            f"column {value!r}: zero total value mass (Lorenz undefined)")
    pop = np.r_[0.0, np.cumsum(ws) / ws.sum()]
    val = np.r_[0.0, np.cumsum(ws * xs) / mass]
    return pd.DataFrame({"population_share": pop, "value_share": val})


def weighted_gini(values, weights=None) -> float:
    """Standard non-negative survey-weighted Gini coefficient.

    Over the sorted non-negative values (the survey convention):
    ``G = sum(w*(2*cw - w - W)*x) / (W*sum(w*x))`` with ``cw`` the
    cumulative weight. Returns NaN when the value mass is zero.
    """
    x = np.asarray(values, dtype=float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0) & (x >= 0)
    x, w = x[ok], w[ok]
    if len(x) == 0 or (w * x).sum() == 0:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    total_w = cw[-1]
    return float((w * (2 * cw - w - total_w) * x).sum()
                 / (total_w * (w * x).sum()))


def styled_lorenz(
    data: pd.DataFrame,
    *,
    value: str,
    weights: str | None = None,
    top_shares: list[float] | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    source: str | None = None,
    figsize: tuple[float, float] | None = None,
    outpath: str | Path | None = None,
    close: bool = False,
    style: dict | None = None,
):
    """A Lorenz curve with 45-degree line and Gini annotation.

    The curve comes from ``lorenz_points``; the Gini is the standard
    non-negative survey-weighted coefficient (``weighted_gini``).
    ``top_shares=[50, 90, 99]`` marks those population percentiles on
    the curve and annotates the share held by the top group. Returns
    ``(fig, ax)``.
    """
    s = {**HOUSE_STYLE, **(style or {})}
    pts = lorenz_points(data, value, weights)
    gini = weighted_gini(
        data[value].to_numpy(float),
        data[weights].to_numpy(float) if weights else None)

    fig, ax = plt.subplots(figsize=figsize or s["figsize"])
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["gray"],
            linewidth=1.2, label="Perfect equality")
    ax.plot(pts["population_share"], pts["value_share"],
            color=s["colors"][0], linewidth=2.0, label="Lorenz curve")
    ax.text(0.04, 0.82, f"Gini = {gini:.3f}", transform=ax.transAxes,
            fontsize=s["label_size"], color=s["colors"][0])
    for p in (top_shares or []):
        # The value share at the p-th population percentile, marked and
        # annotated along the curve (the reading aids).
        idx = int(np.searchsorted(pts["population_share"], p / 100,
                                  side="left"))
        idx = min(idx, len(pts) - 1)
        pop = float(pts["population_share"].iloc[idx])
        val = float(pts["value_share"].iloc[idx])
        ax.plot(pop, val, "o", color=s["colors"][0], markersize=5, zorder=5)
        ax.annotate(f"Top {100 - p:g}%: {100 * (1 - val):.1f}%",
                    (pop, val), xytext=(8, -12), textcoords="offset points",
                    fontsize=s["note_size"], color=PALETTE["gray"])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    # Symmetric scales: the curve must read on a cube-like unit square.
    ax.set_aspect("equal", adjustable="box")
    _cosmetics(ax, s, title, xlabel, ylabel)
    ax.legend(fontsize=s["tick_size"], frameon=False, loc="upper left")
    _finish(fig, s, source, outpath, close=close)
    return fig, ax


def styled_twinaxis(
    data: pd.DataFrame,
    *,
    x: str,
    bar: str,
    line: str,
    bar_label: str = "",
    line_label: str = "",
    align_axes: bool = False,
    title: str = "",
    xlabel: str = "",
    bar_ylabel: str = "",
    line_ylabel: str = "",
    source: str | None = None,
    value_fmt: str = "{:.2f}",
    figsize: tuple[float, float] | None = None,
    outpath: str | Path | None = None,
    close: bool = False,
    style: dict | None = None,
):
    """Bars on the left axis and a line on the right axis.

    ``data`` is tidy: one row per x category; ``bar`` is the left-axis
    series, ``line`` the right-axis one. The two series MUST use two
    distinct strong hues -- the palette rule (enforced here). Missing
    values stay as gaps and are counted via ``info()``. Returns
    ``align_axes=True`` gives both axes the SAME limits -- when the two
    series share a scale (two shares, two rates), one level reads the
    same on both sides. Returns ``(fig, (ax_bar, ax_line))``.
    """
    s = {**HOUSE_STYLE, **(style or {})}
    if s["colors"][0] == s["colors"][1]:
        raise ValueError("bar and line need two distinct strong hues "
                         "(palette rule: co-plotted series differ in hue)")
    labels = [str(v) for v in data[x]]
    centers = np.arange(len(data))
    bar_vals = data[bar].to_numpy(float)
    line_vals = data[line].to_numpy(float)
    for col, vals in ((bar, bar_vals), (line, line_vals)):
        if not np.isfinite(vals).any():
            raise ValueError(f"column {col!r}: all values are missing")
        missing = int((~np.isfinite(vals)).sum())
        if missing:
            info(f"{col}: {missing} missing values left as gaps")

    fig, ax_bar = plt.subplots(figsize=figsize or s["figsize"])
    bars = ax_bar.bar(centers, bar_vals, width=s["bar_width"],
                      color=s["colors"][0], edgecolor="white",
                      label=bar_label)
    if s["value_label"]:
        ax_bar.bar_label(bars, fmt=lambda v: value_fmt.format(v),
                         fontsize=s["note_size"], padding=2)
    ax_bar.set_xticks(centers, labels, fontsize=s["tick_size"])
    _cosmetics(ax_bar, s, title, xlabel, bar_ylabel)
    # The left axis wears its series color, like the right one.
    ax_bar.set_ylabel(bar_ylabel, fontsize=s["label_size"],
                      color=s["colors"][0])
    ax_bar.tick_params(axis="y", labelcolor=s["colors"][0])
    # Bars read from the zero base with headroom for the value labels, on the
    # side the data actually goes (all-negative bars keep their baseline).
    if np.isfinite(bar_vals).any():
        blo, bhi = float(np.nanmin(bar_vals)), float(np.nanmax(bar_vals))
        ax_bar.set_ylim(min(0.0, blo) * 1.25, max(0.0, bhi) * 1.25)

    ax_line = ax_bar.twinx()
    ax_line.plot(centers, line_vals, marker="o", color=s["colors"][1],
                 linewidth=1.8, markersize=4.5, label=line_label)
    ax_line.set_ylabel(line_ylabel, fontsize=s["label_size"],
                       color=s["colors"][1])
    ax_line.tick_params(axis="y", labelsize=s["tick_size"],
                        labelcolor=s["colors"][1])
    ax_line.spines[["top", "left"]].set_visible(False)
    if align_axes:
        # One shared scale across both axes.
        both = np.concatenate([bar_vals, line_vals])
        both = both[np.isfinite(both)]
        lo = min(0.0, float(both.min()))
        hi = float(both.max()) * 1.25
        ax_bar.set_ylim(lo, hi)
        ax_line.set_ylim(lo, hi)
    elif np.isfinite(line_vals).any() and np.nanmin(line_vals) >= 0:
        # Same headroom on the right axis when the series is non-negative.
        ax_line.set_ylim(0, float(np.nanmax(line_vals)) * 1.25)

    # One combined legend on the left axis, colored by each series.
    h1, l1 = ax_bar.get_legend_handles_labels()
    h2, l2 = ax_line.get_legend_handles_labels()
    ax_bar.legend(h1 + h2, l1 + l2, fontsize=s["tick_size"], frameon=False,
                  loc="upper left")
    _finish(fig, s, source, outpath, close=close)
    return fig, (ax_bar, ax_line)


def stackedbar_points(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    absolute: bool = False,
) -> pd.DataFrame:
    """The plotted numbers of ``styled_stackedbar`` as a tidy frame.

    One row per (x, group): ``value`` is the stacked height (missing
    values enter as 0, count reported via ``info()``) and ``share`` the
    within-x percentage (100 percent unless ``absolute=True``, where
    ``share`` mirrors ``value``). Group hues come from the palette.
    """
    if data.empty:
        raise ValueError(f"columns {x!r}/{group!r}: nothing to stack")
    dup = data.duplicated(subset=[x, group], keep=False)
    if dup.any():
        cells = (data.loc[dup, [x, group]].drop_duplicates().head(4)
                 .to_numpy().tolist())
        raise ValueError(
            f"columns {x!r}/{group!r}: {int(dup.sum())} duplicate rows -- "
            f"one row per (x, group) expected, e.g. {cells}; aggregate first")
    # An all-missing series must fail NAMED (pivot_table would silently
    # drop the group column instead).
    for level in pd.unique(data[group]):
        if data.loc[data[group] == level, y].notna().sum() == 0:
            raise ValueError(
                f"column {y!r}: series is all missing for group {level!r}")
    wide = data.pivot_table(index=x, columns=group, values=y, observed=True)
    wide = wide.reindex(data[x].drop_duplicates())
    # Group levels in APPEARANCE order: pivot sorts columns
    # alphabetically, which would hand the prime colors to the wrong
    # groups and restack the legend.
    wide = wide.reindex(columns=list(dict.fromkeys(data[group])))
    vals = wide.to_numpy(float)
    missing = int(np.isnan(vals).sum())
    if missing:
        info(f"{y}: {missing} missing values treated as 0")
    vals = np.nan_to_num(vals)
    totals = vals.sum(axis=1)
    if not absolute and (totals <= 0).any():
        raise ValueError(
            f"column {y!r}: non-positive total in some {x!r} category")
    scale = np.ones(len(totals)) if absolute else 100.0 / totals

    rows = []
    for i, cat in enumerate(wide.index):
        for j, level in enumerate(wide.columns):
            value = float(vals[i, j])
            rows.append({x: cat, group: level, "value": value,
                         "share": value * scale[i]})
    return pd.DataFrame(rows)


def styled_stackedbar(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    absolute: bool = False,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    source: str | None = None,
    figsize: tuple[float, float] | None = None,
    outpath: str | Path | None = None,
    close: bool = False,
    style: dict | None = None,
):
    """Composition of ``y`` across ``group`` levels within each ``x``.

    Normalized to 100 percent by default (``absolute=True`` stacks raw
    values). Group levels get distinct strong hues (palette rule) and
    the legend sits OUTSIDE the axes. Segments carry no value labels by
    design -- compositions are read from the shared 100 percent scale.
    Returns ``(fig, ax)``.
    """
    s = {**HOUSE_STYLE, **(style or {})}
    pts = stackedbar_points(data, x=x, y=y, group=group, absolute=absolute)
    n_levels = len(dict.fromkeys(data[group]))
    if n_levels > len(s["colors"]):
        raise ValueError(
            f"column {group!r}: {n_levels} levels but only "
            f"{len(s['colors'])} distinct strong hues -- raise the palette "
            'via style={"colors": [...]} or facet into panels')
    wide = pts.pivot(index=x, columns=group, values="share")
    wide = wide.loc[data[x].drop_duplicates()]
    wide = wide.reindex(columns=list(dict.fromkeys(data[group])))

    fig, ax = plt.subplots(figsize=figsize or s["figsize"])
    centers = np.arange(len(wide))
    bottom = np.zeros(len(wide))
    for j, level in enumerate(wide.columns):
        heights = wide[level].to_numpy(float)
        ax.bar(centers, heights, width=s["bar_width"], bottom=bottom,
               color=s["colors"][j % len(s["colors"])], label=str(level))
        bottom += heights
    ax.set_xticks(centers, [str(c) for c in wide.index],
                  fontsize=s["tick_size"])
    _cosmetics(ax, s, title, xlabel, ylabel)
    if not absolute:
        ax.set_ylim(0, 100)
    # Legend outside the axes; _finish reserves the right margin for it.
    ax.legend(fontsize=s["tick_size"], frameon=False, bbox_to_anchor=(1.02, 1),
              loc="upper left")
    _finish(fig, s, source, outpath,
            rect=(0, 0.03 if source else 0, 0.80, 1), close=close)
    return fig, ax


def styled_panel(
    panels: list,
    *,
    nrows: int,
    ncols: int,
    titles: str | list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    source: str | None = None,
    outpath: str | Path | None = None,
    close: bool = False,
    style: dict | None = None,
):
    """A grid of related descriptives behind ONE figure.

    ``panels`` holds one entry per cell, in row-major order: either an
    ``f(ax)`` callable (a closure over its data) or a
    ``(draw_fn, data, kwargs)`` tuple called as ``draw_fn(ax, data,
    **kwargs)``. The helper owns only the grid, the title(s) and the
    source note -- each panel keeps its own axes cosmetics. ``titles``
    is either one shared string (figure title) or a list of per-panel
    titles. Returns ``(fig, axes)`` (2D array); the figure is one
    element for ``show_figures``.
    """
    s = {**HOUSE_STYLE, **(style or {})}
    if len(panels) != nrows * ncols:
        raise ValueError(
            f"panels: {len(panels)} entries cannot fill a {nrows}x{ncols} grid")
    cell = s["series_figsize"]
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False,
        figsize=figsize or (cell[0] * ncols, cell[1] * nrows))
    for ax, panel in zip(axes.ravel(), panels):
        if callable(panel):
            panel(ax)
        else:
            draw_fn, pdata, kwargs = panel
            draw_fn(ax, pdata, **kwargs)
    if isinstance(titles, str):
        fig.suptitle(titles, fontsize=s["title_size"])
    elif titles:
        for ax, t in zip(axes.ravel(), titles):
            ax.set_title(t, fontsize=s["title_size"])
    rect = (0, 0.03 if source else 0, 1,
            0.94 if isinstance(titles, str) else 1)
    _finish(fig, s, source, outpath, rect=rect, close=close)
    return fig, axes


# ---------------------------------------------------------------------------
# The figure SERIES: side by side, one scrolling row (diagram analogue of RA)
# ---------------------------------------------------------------------------

def show_figures(
    figures: list,
    *,
    captions: list[str] | None = None,
    img_height: int = 240,
    widths: list[int] | None = None,
    close: bool = False,
) -> None:
    """Display a series of figures SIDE BY SIDE, left to right.

    ``figures`` is a list of matplotlib Figures (or PNG paths); each is
    rendered to PNG bytes and embedded. The row never line-breaks -- it
    scrolls horizontally (``flex-wrap: nowrap`` + ``overflow-x: auto``).
    ``widths`` optionally sets one pixel width per panel (the height then
    follows the aspect ratio), so figures of different shapes -- e.g. a
    2x2 composite next to a single chart -- show at their natural
    proportions without distortion. ``close=True`` closes the Figure
    objects after embedding: pass it whenever figures are produced in a
    LOOP, or the pyplot registry keeps every one of them alive (25 calls =
    25 open figures, and matplotlib's 20-figure warning). ONE output; in a
    script, prints one info line per figure instead.
    """
    if widths is not None and len(widths) != len(figures):
        raise ValueError("widths: need exactly one entry per figure")
    if captions is not None and len(captions) != len(figures):
        raise ValueError("captions: need exactly one entry per figure")
    panels = []
    for i, fig in enumerate(figures):
        if isinstance(fig, (str, Path)):
            png = Path(fig).read_bytes()
            name = str(fig)
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=HOUSE_STYLE["dpi"],
                        bbox_inches="tight")
            png = buf.getvalue()
            name = f"figure {i + 1}"
            if close:
                plt.close(fig)
        if not _in_notebook():
            print(f"figure {i + 1}: {name}")
            continue
        b64 = base64.b64encode(png).decode("ascii")
        cap = escape(captions[i]) if captions else ""
        if widths is not None:
            img_style = f"width: {widths[i]}px; display: block"
        else:
            img_style = f"height: {img_height}px; display: block"
        panels.append(
            '<div style="flex: 0 0 auto; display: flex; flex-direction: '
            'column; gap: 4px">'
            f'<div style="font-family: monospace; font-weight: bold; '
            f'color: {_INFO_COLOR}">{cap}</div>'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="{img_style}">'
            "</div>"
        )
    if not _in_notebook():
        return
    from IPython.display import HTML, display
    display(HTML(
        '<div style="display: flex; gap: 24px; align-items: flex-start; '
        'flex-wrap: nowrap; overflow-x: auto; padding-bottom: 8px">'
        + "".join(panels) + "</div>"
    ))


def write_figure_data(data: pd.DataFrame, outpath: str | Path) -> Path:
    """Persist the exact numbers behind a figure as CSV (reproducibility).

    A figure is only verifiable if the plotted values survive as data;
    save them next to the PNG, same stem, ``_data.csv`` suffix. The tidy
    long CSV is the house DEFAULT form for any output that is not a
    regression table.
    """
    outpath = Path(outpath)
    csv_path = outpath.with_name(outpath.stem + "_data.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=False)
    return csv_path


#: Dark purple: the header marker of every standardized table.


