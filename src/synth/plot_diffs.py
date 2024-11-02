from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_differences(directories):
    from dolphin import io
    from opera_utils import get_dates

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(directories)))

    diffs = []
    errors = []
    dfs = []
    for directory, color in zip(directories, colors):
        # Set up paths
        main_dir = Path(directory)
        diff_dir = main_dir / "differences"

        # Read data
        reader = io.RasterStackReader.from_file_list(sorted(diff_dir.glob("*tif")))
        dates = [get_dates(f)[1] for f in reader.file_list]
        temp_coh = io.load_gdal(main_dir / "interferograms/temporal_coherence.tif")
        sim = io.load_gdal(main_dir / "interferograms/similarity.tif")

        # Process data
        # TODO: process in patches?
        pixels = reader[:, :, :].reshape(reader.shape[0], -1)

        # Calculate product
        # cur_pixel_diffs = np.angle(np.exp(1j * (pixels - p0[:, None])))
        cur_pixel_diffs = pixels

        # Plot on first subplot
        # ax1.plot(dates, cur_pixel_diffs, lw=0.5, alpha=0.2, color=color)

        # Plot on second subplot
        rmse_by_date = rmse(cur_pixel_diffs, axis=1)
        ax2.plot(dates, rmse_by_date, color=color, label=Path(directory).name)

        rmse_by_pixel = rmse(cur_pixel_diffs, axis=0)
        df = pd.DataFrame(
            data={
                "temporal_coherence": temp_coh.ravel(),
                "similarity": sim.ravel(),
                "rmse": rmse_by_pixel,
            }
        )

        errors.append(rmse_by_date)
        diffs.append(cur_pixel_diffs)
        dfs.append(df)

    # Configure first subplot
    ax1.set_title("Errors by date")
    ax1.set_ylabel("Wrapped Error [radians]")

    # Configure second subplot
    ax2.set_ylabel("RMSE [radians]")
    ax2.legend()

    plt.tight_layout()
    plt.show()
    # return errors, diffs
    return errors, dfs


def rmse(arr, axis=1):
    """Calculate the Root Mean Square Error (RMSE) along a specified axis."""
    return np.sqrt(np.mean(arr * arr.conj(), axis=axis))


def plot_temporal_coherence_vs_rmse(df):
    """Create a comprehensive visualization of temporal coherence versus RMSE relationships.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least two columns:
        - 'temporal_coherence': temporal coherence values
        - 'rmse': RMSE values

    Notes
    -----
    Creates a figure with four subplots:
    1. Scatter plot of temporal coherence vs RMSE with color gradient
    2. Box plot showing RMSE distribution across temporal coherence bins
    3. Violin plot showing RMSE distribution across temporal coherence bins
    4. Heatmap showing density of temporal coherence vs RMSE values

    The temporal coherence values are binned into 5 categories:
    0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, and 0.8-1.0

    The visualization uses seaborn for enhanced statistical plotting.
    """
    # Create bins for temporal coherence
    df["coherence_bin"] = pd.cut(
        df["temporal_coherence"],
        bins=5,
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
    )

    # Set up the matplotlib figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Temporal Coherence vs RMSE Analysis", fontsize=16)

    # 1. Scatter plot with color gradient
    sns.scatterplot(
        data=df,
        x="temporal_coherence",
        y="rmse",
        hue="temporal_coherence",
        palette="viridis",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Temporal Coherence vs RMSE")

    # 2. Box plot for each bin
    sns.boxplot(data=df, x="coherence_bin", y="rmse", ax=axes[0, 1])
    axes[0, 1].set_title("RMSE for Temporal Coherence Bins")
    axes[0, 1].set_xlabel("Temporal Coherence Bin")

    # 3. Violin plot for each bin
    sns.violinplot(data=df, x="coherence_bin", y="rmse", ax=axes[1, 0])
    axes[1, 0].set_title("RMSE Distribution for Temporal Coherence Bins")
    axes[1, 0].set_xlabel("Temporal Coherence Bin")

    # 4. Heatmap of average RMSE for coherence-rmse grid
    heatmap_data = df.pivot_table(
        values="rmse",
        index=pd.cut(df["temporal_coherence"], bins=10),
        columns=pd.cut(df["rmse"], bins=10),
        aggfunc="count",
    )
    sns.heatmap(heatmap_data, ax=axes[1, 1], cmap="YlOrRd")
    axes[1, 1].set_title("Density of Temporal Coherence vs RMSE")
    axes[1, 1].set_xlabel("RMSE Bins")
    axes[1, 1].set_ylabel("Temporal Coherence Bins")

    plt.tight_layout()
    plt.show()


def similarity_vs_temporal_coherence(df):
    """Create hexbin plots comparing similarity and temporal coherence against RMSE.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing three columns:
        - 'temporal_coherence': temporal coherence values
        - 'similarity': cosine similarity values
        - 'rmse': RMSE values

    Notes
    -----
    Creates a figure with two hexbin plots:
    1. Temporal coherence vs RMSE
    2. Similarity vs RMSE

    The hexbin plots use logarithmic binning and a viridis colormap to show
    density of points. The similarity plot is constrained to the range [-0.1, 1.1]
    for better visualization.
    """
    # Set up the matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(9, 6), squeeze=False)
    fig.suptitle("Similarity vs Temporal Coherence - RMSE Analysis", fontsize=16)

    # Scatter plots with color gradient
    ax = axes[0, 0]
    grid_kw = {
        "gridsize": [20, 20],
        "cmap": "viridis",
        "mincnt": 10,
        "bins": "log",
    }
    ax.hexbin(df["temporal_coherence"], df["rmse"], **grid_kw)
    ax.set_xlabel("temporal_coherence")
    ax.set_ylabel("rmse")

    ax = axes[0, 1]
    grid_kw["gridsize"][0] *= 2
    ax.hexbin(df["similarity"], df["rmse"], **grid_kw)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("RMSE")
    ax.set_xlim(-0.1, 1.1)
    fig.tight_layout()
