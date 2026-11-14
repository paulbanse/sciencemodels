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
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# legacy-column shim: old report CSVs -> current report schema
LEGACY_COLUMN_RENAMES = {
    "harvested_50_step": "harvested_10_step",
    "harvested_90_step": "harvested_25_step",
}
_REVERSE_COLUMN_RENAMES = {new: old for old, new in LEGACY_COLUMN_RENAMES.items()}


def _apply_legacy_shim(df):
    """Rename legacy report columns to their current names (returns df unchanged if none)."""
    renames = {
        old: new
        for old, new in LEGACY_COLUMN_RENAMES.items()
        if old in df.columns and new not in df.columns
    }
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


def plot_avg_time_vs_corr_prest_merit(foldername, curiosity,ax =None, length=401):

    corrs =  [[] for k in range(length)]
    
    for filename in os.listdir(foldername):
        if  f"curiosity{curiosity}" in filename:
            df = read_csv(os.path.join(foldername, filename))
            for step, corr in zip(df['Step'], df['corr_prestige_localMerit']):
                corrs[step].append(corr)

    mean_corrs = [np.mean(corrs_at_t) for corrs_at_t in corrs]
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
        plot_avg_time_vs_corr_prest_merit(foldername, curiosity, ax=ax,length=1601)
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
