# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch
import matplotlib.ticker as mtick


# Load the data
def load_data(file_path):
    """
    Load CSV data into a pandas DataFrame with appropriate types.

    Parameters:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    df = pd.read_csv(file_path)

    # Convert columns to appropriate types
    for col in ["Epipolar", "Fundamental"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN values in key metrics
    df = df.dropna(subset=["Epipolar", "Fundamental"])

    return df


def create_boxplot_figure(
    df,
    metric,
    output_filename,
    title=None,
    y_label=None,
    log_scale=False,
    fig_width=10,
    fig_height=6,
    custom_order=None,
    highlight_methods=None,
):
    """
    Create a publication-quality boxplot for the specified metric.

    Parameters:
        df (pd.DataFrame): Data frame containing the data
        metric (str): Column name for the metric to plot
        output_filename (str): Filename for saving the figure
        title (str, optional): Plot title
        y_label (str, optional): Y-axis label
        log_scale (bool): Use log scale for y-axis if True
        fig_width (float): Figure width in inches
        fig_height (float): Figure height in inches
        custom_order (list): Custom order of categories for x-axis
        highlight_methods (list): List of method names to highlight

    Returns:
        fig, ax: The created figure and axis objects
    """
    # Set style for scientific publication
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            # Use only common fonts guaranteed to be available
            "font.serif": [
                "DejaVu Serif",
                "Bitstream Vera Serif",
                "Liberation Serif",
                "Times New Roman",
            ],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Determine the order of methods on x-axis
    if custom_order is None:
        # Default order: alphabetical with "OURS" methods at the end
        all_methods = df["name"].unique().tolist()
        ours_methods = [m for m in all_methods if m.startswith("OURS")]
        other_methods = [m for m in all_methods if not m.startswith("OURS")]
        other_methods.sort()  # Sort alphabetically
        ours_methods.sort()  # Sort OURS methods
        method_order = other_methods + ours_methods
    else:
        method_order = custom_order

    # Create a temporary column for color assignment
    df_plot = df.copy()
    df_plot["highlight"] = df_plot["name"].apply(
        lambda x: "Highlighted" if x in highlight_methods else "Regular"
    )

    # Create color mapping dictionary
    hue_order = ["Regular", "Highlighted"]
    palette_dict = {
        "Regular": "#4477AA",  # Default blue
        "Highlighted": "#FF5555",  # Highlighted in red
    }

    # Create the boxplot with seaborn - fixed for deprecation warning
    sns.boxplot(
        x="name",
        y=metric,
        hue="highlight",  # Use hue instead of palette
        hue_order=hue_order,
        palette=palette_dict,
        data=df_plot,
        order=method_order,
        ax=ax,
        width=0.6,
        fliersize=3,
        linewidth=1.25,
        legend=False,  # Hide the legend as recommended
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "gray",
            "alpha": 0.5,
        },
    )

    # Add individual data points with jitter for better visualization
    sns.stripplot(
        x="name",
        y=metric,
        data=df_plot,
        order=method_order,
        ax=ax,
        color="black",
        size=1.5,
        alpha=0.2,
        jitter=0.2,
        dodge=False,
        linewidth=0,
    )

    # Apply log scale if requested
    if log_scale:
        ax.set_yscale("log")
        # Configure log scale formatting
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)

    # Set labels and title
    if y_label:
        ax.set_ylabel(y_label, fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold")

    # Format x-axis labels
    plt.xticks(rotation=45, ha="right")
    ax.set_xlabel("")  # Remove x-axis label

    # Make OURS labels bold and red
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):
        method = method_order[i]
        if highlight_methods and method in highlight_methods:
            label.set_fontweight("bold")
            label.set_color("#FF0000")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add grid for better readability (y-axis only)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Improve margins
    plt.tight_layout()

    # Save figure
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)

    return fig, ax


# Load data
df = load_data("results.csv")

# Define methods to highlight (OURS methods)
ours_methods = [method for method in df["name"].unique() if "OURS" in method]

# Custom order: alphabetical with OURS at the end
all_methods = df["name"].unique().tolist()
other_methods = [m for m in all_methods if "OURS" not in m]
other_methods.sort()  # Sort alphabetically
ours_methods.sort()  # Sort OURS methods
method_order = other_methods + ours_methods

# Create boxplot for Epipolar metric
create_boxplot_figure(
    df=df,
    metric="Epipolar",
    output_filename="epipolar_boxplot.png",
    title="Epipolar Error Comparison Across Methods",
    y_label="Epipolar Error",
    log_scale=True,  # Log scale for better visualization of range
    fig_width=12,
    fig_height=7,
    custom_order=method_order,
    highlight_methods=ours_methods,
)

# Create boxplot for Fundamental metric
create_boxplot_figure(
    df=df,
    metric="Fundamental",
    output_filename="fundamental_boxplot.png",
    title="Fundamental Matrix Error Comparison Across Methods",
    y_label="Fundamental Matrix Error",
    log_scale=True,  # Log scale for better visualization of range
    fig_width=12,
    fig_height=7,
    custom_order=method_order,
    highlight_methods=ours_methods,
)

print("Boxplots generated successfully:")
print("1. epipolar_boxplot.png")
print("2. fundamental_boxplot.png")
