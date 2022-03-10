from math import ceil
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import cm


def plot_losses(
    filepath: Path, losses: np.ndarray, val_losses: np.ndarray, val_every: int = 3
):
    f, ax = plt.subplots(figsize=(12, 4))
    ax.plot(losses, label="Training")
    # plot validation losses
    X = range(0, len(val_losses) * val_every, val_every)
    ax.plot(X, val_losses, label="Validation")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Epoch")
    ax.set_xlabel("Loss")
    ax.legend()
    f.savefig(filepath / "losses.png")
    plt.close("all")


def plot_horizon_losses(
    filepath: Path,
    error: xr.DataArray,
    identifier: Optional[str] = None,
    site_dim: Optional[str] = "site",
):
    # make the forecast horizon plot
    f, ax = plt.subplots(figsize=(12, 6))
    for site in error[site_dim].values:
        error.sel({site_dim: site}).squeeze().plot(ax=ax, label=site)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel(error.name if error.name is not None else "Error")
    ax.set_xlabel("Horizon")
    ax.set_title("Performance over forecast horizon")
    ax.legend()

    f.savefig(
        filepath / "horizon_losses.png"
        if identifier is None
        else filepath / f"{identifier}_horizon_losses.png"
    )
    plt.close("all")


def plot_timeseries_over_horizon(
    filepath: Path, preds: xr.Dataset, site_dim: Optional[str] = "site"
):
    for site in np.unique(preds[site_dim].values):
        # make the timeseries plots
        preds_ = preds.sel({site_dim: site})
        f, axs = plt.subplots(
            3, 4, figsize=(6 * 4, 2 * 3), tight_layout=True, sharey=True, sharex=True
        )
        f.tight_layout(rect=[0, 0.03, 1, 0.95])

        random_times = np.random.choice(preds_["date"].values, size=12, replace=False)

        for ix, time in enumerate(random_times):
            ax = axs[np.unravel_index(ix, (3, 4))]
            ax.plot(preds_.sel({"date": time})["obs"], label="obs")
            ax.plot(preds_.sel({"date": time})["sim"], label="sim")
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

            ax.set_title(time)

        ax.legend()

        f.suptitle(f"{site} Timeseries")
        f.savefig(filepath / f"{site}_demo_timeseries.png")
        plt.close("all")


def plot_test_preds(
    filepath: Path,
    preds: xr.Dataset,
    test_chunks: List[List[str]],
    site_dim: Optional[str] = "site",
):
    cmap = cm.get_cmap("winter_r")
    n_cols = 3
    n_rows = ceil(len(preds["site"]) / n_cols) * len(test_chunks)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows))
    axs = axs.flatten()
    _ii = 0
    for chunk in test_chunks:
        date_idx = (preds["date"] >= pd.to_datetime(chunk[0])) & (
            preds["date"] <= pd.to_datetime(chunk[1])
        )
        for site in preds["site"].data:

            for step in [0, 5, 10, 15, 25, 50, 75, 89]:
                # print (site, step)
                hexcolor = "#" + "".join(
                    f"{int(el*255):02x}" for el in cmap(int(step / 90 * 255))[:3]
                )
                preds.sel({"site": site, "step": step, "date": date_idx}).shift(
                    date=step
                )["sim"].plot(ax=axs[_ii], c=hexcolor)

            preds.sel({"site": site, "step": 0, "date": date_idx})["obs"].plot(
                ax=axs[_ii], c="#f200ff"
            )
            _ii += 1

    fig.savefig(filepath / "test_timeseries_allsites.png")
    plt.close("all")
