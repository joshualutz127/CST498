from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

# Locate files in the same folder as this Python script.
BASE_DIR = Path(__file__).resolve().parent

# This file is created from the real FiveThirtyEight NBA data
# by prepare_basketball_games() in the main project pipeline.
DATA_PATH = BASE_DIR / "basketball_games.csv"

# Use the ten most recent seasons available in this dataset.
MIN_SEASON = 2009


# ---------------------------------------------------------------
# 1. Load and validate real project data
# ---------------------------------------------------------------

def load_project_data(path: Path) -> pd.DataFrame:
    """
    Load the real prepared NBA game dataset used by the
    capstone machine-learning pipeline.

    Required columns:
    season, game_date, target, elo1_pre, elo2_pre,
    elo_prob1, and elo_diff
    """

    # Stop if the real prepared dataset is not available.
    if not path.exists():
        raise FileNotFoundError(
            f"\nReal NBA dataset not found at:\n{path}\n\n"
            "Run the main machine-learning pipeline first. "
            "That program downloads the real FiveThirtyEight "
            "NBA data and creates basketball_games.csv."
        )

    df = pd.read_csv(path)

    required_columns = {
        "season",
        "game_date",
        "target",
        "elo1_pre",
        "elo2_pre",
        "elo_prob1",
        "elo_diff",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "The dataset does not contain all required columns. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    # Convert the dates into valid datetime values.
    df["game_date"] = pd.to_datetime(
        df["game_date"],
        errors="coerce"
    )

    if df["game_date"].isna().any():
        invalid_dates = int(df["game_date"].isna().sum())

        raise ValueError(
            f"The dataset contains {invalid_dates} invalid dates."
        )

    # Confirm that target contains only the expected values.
    valid_target_values = set(df["target"].dropna().unique())

    if not valid_target_values.issubset({0, 1}):
        raise ValueError(
            "The target column must contain only 0 and 1."
        )

    # Keep the 2009-2018 seasons for the visualizations.
    df = df[df["season"] >= MIN_SEASON].copy()
    df = df.sort_values("game_date").reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No games were found from season {MIN_SEASON} or later."
        )

    print("\nLoaded real historical NBA data.")
    print(f"Dataset: {path}")
    print(f"Number of games: {len(df):,}")
    print(
        f"Seasons: {df['season'].min()}-"
        f"{df['season'].max()}"
    )

    return df


# ---------------------------------------------------------------
# 2. Correlation matrix and most highly correlated pair
# ---------------------------------------------------------------

def plot_correlation_and_top_pair(
    df: pd.DataFrame
) -> None:
    """
    Create a correlation matrix and a scatter plot of the
    two variables with the strongest absolute correlation.
    """

    numeric_columns = [
        "target",
        "elo1_pre",
        "elo2_pre",
        "elo_prob1",
        "elo_diff",
    ]

    correlation_matrix = (
        df[numeric_columns]
        .corr()
        .round(2)
    )

    # Figure 1: Correlation matrix heatmap
    plt.figure(figsize=(7, 5.5))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.title("Correlation Matrix of Pregame Variables")
    plt.tight_layout()

    figure_1_path = (
        BASE_DIR / "fig1_correlation_matrix.png"
    )

    plt.savefig(figure_1_path, dpi=150)
    plt.close()

    print(f"Saved {figure_1_path.name}")

    # Find the two different columns with the highest
    # absolute correlation.
    maximum_correlation = 0
    column_a = None
    column_b = None

    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            first_column = numeric_columns[i]
            second_column = numeric_columns[j]

            correlation = abs(
                correlation_matrix.loc[
                    first_column,
                    second_column
                ]
            )

            if correlation > maximum_correlation:
                maximum_correlation = correlation
                column_a = first_column
                column_b = second_column

    actual_correlation = correlation_matrix.loc[
        column_a,
        column_b
    ]

    print(
        f"Most correlated pair: {column_a} and "
        f"{column_b} (r = {actual_correlation})"
    )

    # Figure 2: Scatter plot of the most correlated pair
    plt.figure(figsize=(7, 5))

    sns.scatterplot(
        data=df,
        x=column_a,
        y=column_b,
        hue="target",
        alpha=0.6,
        palette="Set1"
    )

    plt.title(
        f"{column_a} vs. {column_b} "
        f"(r = {actual_correlation:.2f})"
    )

    plt.xlabel(column_a)
    plt.ylabel(column_b)
    plt.legend(title="Team 1 Win (target)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    figure_2_path = (
        BASE_DIR / "fig2_top_correlated_scatter.png"
    )

    plt.savefig(figure_2_path, dpi=150)
    plt.close()

    print(f"Saved {figure_2_path.name}")


# ---------------------------------------------------------------
# 3. Single-column distribution plot
# ---------------------------------------------------------------

def plot_single_column_distribution(
    df: pd.DataFrame,
    column: str = "elo_diff"
) -> None:
    """
    Plot the distribution of one numeric project variable.
    """

    plt.figure(figsize=(7, 5))

    sns.histplot(
        data=df,
        x=column,
        kde=True,
        color="steelblue"
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Number of Games")
    plt.tight_layout()

    figure_3_path = BASE_DIR / "fig3_distribution.png"

    plt.savefig(figure_3_path, dpi=150)
    plt.close()

    print(f"Saved {figure_3_path.name}")


# ---------------------------------------------------------------
# 4. Bar plot of win rate by Elo-difference range
# ---------------------------------------------------------------

def plot_win_rate_by_elo_bucket(
    df: pd.DataFrame
) -> None:
    """
    Group games into Elo-difference ranges and calculate
    Team 1's actual win rate within each range.
    """

    bucketed_data = df.copy()

    bucketed_data["elo_diff_bucket"] = pd.cut(
        bucketed_data["elo_diff"],
        bins=[
            float("-inf"),
            -150,
            -50,
            50,
            150,
            float("inf"),
        ],
        labels=[
            "Below -150",
            "-150 to -50",
            "-50 to 50",
            "50 to 150",
            "Above 150",
        ]
    )

    win_rate = (
        bucketed_data
        .groupby(
            "elo_diff_bucket",
            observed=True
        )["target"]
        .mean()
    )

    plt.figure(figsize=(8, 5))

    axis = win_rate.plot(
        kind="bar",
        color="darkorange",
        edgecolor="black"
    )

    axis.set_ylabel("Team 1 Win Rate")
    axis.set_xlabel("Pregame Elo-Difference Range")

    axis.set_title(
        "Team 1 Win Rate by Pregame Elo Differential"
    )

    plt.xticks(rotation=0)
    plt.tight_layout()

    figure_4_path = (
        BASE_DIR / "fig4_win_rate_by_elo_bucket.png"
    )

    plt.savefig(figure_4_path, dpi=150)
    plt.close()

    print(f"Saved {figure_4_path.name}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    """
    Load the real project data and generate all four figures.
    """

    project_data = load_project_data(DATA_PATH)

    plot_correlation_and_top_pair(project_data)

    plot_single_column_distribution(
        project_data,
        column="elo_diff"
    )

    plot_win_rate_by_elo_bucket(project_data)

    print("\nAll figures generated using real NBA data.")


if __name__ == "__main__":
    main()
