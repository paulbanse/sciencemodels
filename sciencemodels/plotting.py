"""Shared analysis plotting helper for all notebooks (ADR-0002).

Before the restructure each notebook copy-pasted its own ``plot_smt`` (five
copies across analyze_data.ipynb and the four compare_landscapes_*
notebooks) and longitudinalcomparisions.ipynb kept its own sibling helpers.
All of them live here now; the notebooks import from this module.

The ``read_csv`` wrapper carries the legacy-column shim: report CSVs written
before the ``harvested_*_step`` rename (everything currently under ``data/``)
still plot, because ``harvested_50_step``/``harvested_90_step`` columns are
renamed to ``harvested_10_step``/``harvested_25_step`` on read, and
``plot_smt`` resolves legacy metric names passed at the call sites against
either spelling.

The correlation family drifted three times over, so the shim covers it too:
the current report CSVs carry ``Mean_of_Corr_prestige_local_merit`` (snake
case, ``Mean_of_`` prefix from the datacollector), the live notebook call
sites ask for ``Mean_of_Corr_prestige_localMerit`` (camelCase), and the legacy
CSVs under ``data/`` carry ``corr_prestige_localMerit`` and
``mean_corr_prestige_localMerit``.  Both directions are handled:
``_apply_legacy_shim`` rewrites any legacy spelling to the current schema
name on read, and ``_resolve_column`` maps a requested name onto whichever
spelling a frame actually carries.  ``CORR_METRIC_ALIASES`` is the alias table
for the six metrics; :func:`_canonical_corr_name` is the matching rule, so the
three spellings are interchangeable at every call site.
"""
from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# legacy-column shim: old report CSVs -> current report schema
LEGACY_COLUMN_RENAMES = {
    "harvested_50_step": "harvested_10_step",
    "harvested_90_step": "harvested_25_step",
}
_REVERSE_COLUMN_RENAMES = {new: old for old, new in LEGACY_COLUMN_RENAMES.items()}

# correlation family: canonical suffix after ``Corr_``; the current schema uses
# snake case and an explicit ``Mean_of_`` prefix only for aggregated report rows
CORR_METRIC_SUFFIXES = (
    "prestige_local_merit",
    "prestige_ideal_merit",
    "prestige_visibility_local_merit",
    "prestige_visibility_ideal_merit",
    "prestige_vanishing_local_merit",
    "prestige_vanishing_ideal_merit",
)


def _camel(suffix):
    """``prestige_local_merit`` -> ``prestigeLocalMerit`` (legacy camelCase)."""
    head, *tail = suffix.split("_")
    return head + "".join(part.capitalize() for part in tail)


#: accepted spellings per correlation metric, keyed by canonical name.  The
#: first two entries are the current schema (per-step frame, report CSV).
CORR_METRIC_ALIASES = {
    "corr_" + suffix: (
        "Corr_" + suffix,                  # current per-step frame
        "Mean_of_Corr_" + suffix,          # current report CSV
        "Corr_" + _camel(suffix),          # legacy camelCase frame
        "Mean_of_Corr_" + _camel(suffix),  # legacy notebook call site
        "corr_" + _camel(suffix),          # legacy frame, lowercase corr
        "mean_corr_" + _camel(suffix),     # legacy report CSV
    )
    for suffix in CORR_METRIC_SUFFIXES
}

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# order matters: strip the longest aggregation prefix first
_AGGREGATED_PREFIXES = ("Mean_of_", "mean_of_", "Mean_", "mean_")


def _canonical_corr_name(name):
    """Canonical key of a correlation-family column, or None if outside the family.

    Drops an aggregation prefix (``Mean_of_``/``mean_``), splits camelCase and
    lowercases, so all three spellings of a metric share one key.  Names
    outside the family return None and are never renamed.
    """
    key = name
    for prefix in _AGGREGATED_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    key = _CAMEL_BOUNDARY.sub("_", key).lower()
    return key if key in CORR_METRIC_ALIASES else None


def _is_aggregated(name):
    """True for the aggregated (``Mean_of_``) column of a metric family."""
    return name.startswith(_AGGREGATED_PREFIXES)


def _apply_legacy_shim(df):
    """Rename legacy report columns to their current names (returns df unchanged if none)."""
    renames = {
        old: new
        for old, new in LEGACY_COLUMN_RENAMES.items()
        if old in df.columns and new not in df.columns
    }
    # correlation family: legacy spelling -> current schema name, keeping the
    # aggregation prefix (Mean_of_ report column vs per-step model frame)
    for col in df.columns:
        key = _canonical_corr_name(col)
        if key is None:
            continue
        target = ("Mean_of_" if _is_aggregated(col) else "") + "Corr_" + key[len("corr_"):]
        if col != target and target not in df.columns:
            renames[col] = target
    return df.rename(columns=renames) if renames else df


def read_csv(path):
    """pd.read_csv with the legacy-column shim, so legacy data/ CSVs still plot."""
    return _apply_legacy_shim(pd.read_csv(path))


def _resolve_column(df, name):
    """Map a requested metric name onto the spelling actually present in df."""
    if name in df.columns:
        return name
    for candidate in (
        LEGACY_COLUMN_RENAMES.get(name),
        _REVERSE_COLUMN_RENAMES.get(name),
    ):
        if candidate is not None and candidate in df.columns:
            return candidate
    key = _canonical_corr_name(name)
    if key is not None:
        candidates = [col for col in df.columns if _canonical_corr_name(col) == key]
        # The aggregation class is part of the metric's identity: a per-step
        # request must never take the run-mean column (a mean plotted as a time
        # series is a wrong figure), and a mean request must not take a per-step
        # series. Unresolvable within the class -> return the name unchanged, so
        # the caller raises KeyError instead of plotting the wrong series.
        preferred = [col for col in candidates if _is_aggregated(col) == _is_aggregated(name)]
        if preferred:
            return preferred[0]
    return name


def plot_smt(name, datasets_dict):
    """
    datasets_dict: dictionary with {label: dataframe} pairs
    """
    datasets_dict = {label: _apply_legacy_shim(df) for label, df in datasets_dict.items()}
    curiosities = sorted(list(set().union(*[df["initial_curiosity"].unique() for df in datasets_dict.values()])))
    
    for label, df in datasets_dict.items():
        col = _resolve_column(df, name)
        avg_values = [df[df["initial_curiosity"]==c][col].mean() for c in curiosities]
        std_values = [df[df["initial_curiosity"]==c][col].std() for c in curiosities]
        p_temp = plt.errorbar(curiosities, avg_values, yerr=std_values, fmt='-o', label=label)
        if np.isnan(avg_values[0]):
            plt.hlines(avg_values[5], xmin = 0, xmax = 1,color = p_temp[0].get_color(), linestyles=":")
    
    plt.xlabel("Initial Curiosity")
    plt.ylabel(name)
    plt.title(f"{name} vs Initial Curiosity")
    plt.legend()
    plt.grid()
    plt.show()


def plot_avg_time_vs_corr_prest_merit(foldername, curiosity,ax =None, length=None):
    """Mean correlation over time for one curiosity level.

    ``length`` defaults to None, i.e. derived from the frames actually read: a
    hard-coded horizon longer than the data pads the mean series with NaN and
    floods numpy with "Mean of empty slice" warnings (1601 vs 201 steps on the
    fresh longitudinal companion frames emitted 30,803 of them), so the horizon
    is measured, not assumed.  An explicit ``length`` is still honoured.
    """

    corrs = {}

    for filename in os.listdir(foldername):
        if  f"curiosity{curiosity}" in filename:
            df = read_csv(os.path.join(foldername, filename))
            # current frames carry Corr_prestige_local_merit, legacy frames
            # corr_prestige_localMerit; the shim resolves either spelling
            corr_col = _resolve_column(df, "Corr_prestige_local_merit")
            for step, corr in zip(df['Step'], df[corr_col]):
                corrs.setdefault(int(step), []).append(corr)

    if length is None:
        length = (max(corrs) + 1) if corrs else 0
    mean_corrs = [np.mean(corrs[t]) if t in corrs else np.nan for t in range(length)]
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.plot(range(length), mean_corrs, label=f"Curiosity={curiosity}", alpha=0.7,color = plt.cm.viridis(curiosity))
    ax.set_xlabel("Time")
    ax.set_ylabel("Correlation between Prestige and Merit")


def plot_all_curiosities(foldername):
    fig, ax = plt.subplots(figsize=(10, 6))
    names  = os.listdir(foldername)
    curiosities = [float(name.split("curiosity")[1].split("_")[0]) for name in names if "curiosity" in name]
    unique_curiosities = list(set(curiosities))
    unique_curiosities.sort()
    for curiosity in unique_curiosities:
        plot_avg_time_vs_corr_prest_merit(foldername, curiosity, ax=ax)
    ax.legend()
    plt.show()


def plot_agent_based_data(csv_name,prestige_type="prestige", ax = None):
    df = read_csv(csv_name)
    if ax ==None:
        fig, ax = plt.subplots(figsize=(10, 6))
    for agent_id, agent_data in df.groupby('AgentID'):
        ax.plot(agent_data['Step'], agent_data[prestige_type], alpha=0.7, color = plt.cm.viridis(agent_data["curiosity"].iloc[0]))
    ax.set_xlabel("Time")
    ax.set_ylabel(prestige_type)
    title = csv_name.split("_")[-1].split(".csv")[0]
    ax.set_title(title)
    #ax.set_title(f"{pretige_type} over Time for Agents in {csv_name}")
    # add a colorbar for curiosity


def plot_multiple_agent_based_data(list_of_csv_names, prestige_type="prestige"):
    n = len(list_of_csv_names)
    fig, axs = plt.subplots(1,n, figsize=(n*5, 6), gridspec_kw={'wspace': 0.3})
    for k in range(n):
        ax = axs[k]
        csv_name = list_of_csv_names[k]
        plot_agent_based_data(csv_name, ax =ax, prestige_type=prestige_type)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = plt.colorbar(sm, ax=axs)
    cbar.set_label('Curiosity of the agent')
    plt.show()
