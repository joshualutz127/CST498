"""
CST 499 - Data Visualization Assignment (Revised)
NBA Game Outcome Prediction - Senior Capstone Project

Reads the raw FiveThirtyEight NBA Elo file (basketball_games_raw_538.csv),
which is downloaded by the project's main pipeline, and produces:

  Figure 1  Correlation matrix of the raw game fields
  Figure 2  Team-season average pregame Elo vs. win percentage
  Figure 3  elo_prob1 over the score ratio score1/score2
  Figure 4  3D view: elo_prob1 over score1 and score2
  Figure 5  score1 vs. score2, colored by elo_prob1
  Figure 6  Distributions of elo_diff and home point margin
  Figure 7  Confusion matrices for the training and validation sets

It also prints a prediction performance report (accuracy and confusion
matrix) for a logistic regression model trained on pregame features.

Run:  python nba_data_visualization.py
The CSV file must be in the same folder as this script.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3D)
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "basketball_games_raw_538.csv"

MIN_SEASON = 2014          # ten most recent seasons, 2014-2023
MAX_SEASON = 2023
TRAIN_LAST_SEASON = 2021   # train: 2014-2021, validation: 2022-2023
FORM_WINDOW = 10           # games used for recent-form features
HOME_COURT_ELO = 100       # FiveThirtyEight home-court adjustment

sns.set_theme(style="whitegrid")


# ----------------------------------------------------------------------
# 1. Load the raw FiveThirtyEight file
# ----------------------------------------------------------------------
def load_raw_data(path: Path) -> pd.DataFrame:
    """Load basketball_games_raw_538.csv and add derived columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw NBA file not found: {path}\n"
            "Run the main pipeline first; it downloads "
            "basketball_games_raw_538.csv from FiveThirtyEight."
        )

    df = pd.read_csv(path)

    required = {"date", "season", "team1", "team2", "elo1_pre",
                "elo2_pre", "elo_prob1", "elo1_post", "elo2_post",
                "score1", "score2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Some mirrors store each game twice (once per team).
    # Keep one row per game with team1 as the home team.
    if "is_home" in df.columns:
        neutral = (df["neutral"] == 1) if "neutral" in df.columns else False
        df = df[(df["is_home"] == 1)
                | (neutral & (df["team1"] < df["team2"]))]

    # Remove games that have not been played (no final score).
    df = df.dropna(subset=["score1", "score2"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Derived fields
    df["elo_diff"] = df["elo1_pre"] - df["elo2_pre"]
    df["score_ratio"] = df["score1"] / df["score2"]     # score1/score2
    df["point_margin"] = df["score1"] - df["score2"]
    df["total_points"] = df["score1"] + df["score2"]
    df["home_win"] = (df["score1"] > df["score2"]).astype(int)

    return df


# ----------------------------------------------------------------------
# 2. Pregame recent-form features (no future information)
# ----------------------------------------------------------------------
def add_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, average point margin over its previous FORM_WINDOW
    games in the same season. shift(1) ensures the current game is
    never included, so the feature is known before tipoff.
    """
    home = df[["date", "season", "team1", "point_margin"]].rename(
        columns={"team1": "team", "point_margin": "margin"})
    away = df[["date", "season", "team2", "point_margin"]].rename(
        columns={"team2": "team", "point_margin": "margin"})
    away["margin"] = -away["margin"]
    home["game_idx"] = df.index
    away["game_idx"] = df.index
    home["side"] = 1
    away["side"] = 2

    log = pd.concat([home, away]).sort_values(["date", "game_idx"])
    log["form"] = (
        log.groupby(["team", "season"])["margin"]
        .transform(lambda s: s.shift(1)
                   .rolling(FORM_WINDOW, min_periods=1).mean())
        .fillna(0.0)
    )

    df = df.copy()
    df["form1"] = log[log["side"] == 1].set_index("game_idx")["form"]
    df["form2"] = log[log["side"] == 2].set_index("game_idx")["form"]
    df["form_diff"] = df["form1"] - df["form2"]
    return df


# ----------------------------------------------------------------------
# 3. Check the Elo probability formula
# ----------------------------------------------------------------------
def check_elo_formula(df: pd.DataFrame) -> None:
    """
    elo_prob1 is computed from elo_diff by formula, so the pair is
    not a data finding. FiveThirtyEight adds a 100-point home-court
    adjustment for non-neutral games.
    """
    if "neutral" in df.columns:
        home_games = df[df["neutral"] == 0]
    else:
        home_games = df
    predicted = 1 / (1 + 10 ** (-(home_games["elo_diff"]
                                  + HOME_COURT_ELO) / 400))
    max_error = (home_games["elo_prob1"] - predicted).abs().max()
    print("elo_prob1 = 1 / (1 + 10 ** (-(elo_diff + 100) / 400))")
    print(f"  maximum difference from the file: {max_error:.2e}")


# ----------------------------------------------------------------------
# 4. Figures
# ----------------------------------------------------------------------
def save(fig_name: str) -> None:
    plt.tight_layout()
    plt.savefig(BASE_DIR / fig_name, dpi=150)
    plt.close()
    print(f"Saved {fig_name}")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Figure 1: correlation matrix of the raw and derived fields."""
    columns = ["elo1_pre", "elo2_pre", "elo_diff", "elo_prob1",
               "elo1_post", "elo2_post", "score1", "score2",
               "score_ratio", "point_margin", "home_win"]
    corr = df[columns].corr().round(2)

    plt.figure(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, annot_kws={"size": 8})
    plt.title("Correlation Matrix of basketball_games_raw_538.csv Fields")
    save("fig1_correlation_matrix.png")

    # List the strongest pairs so they can be discussed in the report.
    pairs = (corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
             .stack().abs().sort_values(ascending=False))
    print("\nStrongest correlations (absolute value):")
    print(pairs.head(8).to_string())


def build_team_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine game rows into one row per team per regular season.
    Each game appears twice: once for the home team, once for the
    away team. Adds more fields: win_pct, avg_pregame_elo,
    avg_margin, points_for and points_against.
    """
    if "playoff" in df.columns:
        df = df[df["playoff"].isna()]          # regular season only

    home = pd.DataFrame({
        "season": df["season"], "team": df["team1"],
        "elo_pre": df["elo1_pre"], "win": df["home_win"],
        "margin": df["point_margin"],
        "points_for": df["score1"], "points_against": df["score2"]})
    away = pd.DataFrame({
        "season": df["season"], "team": df["team2"],
        "elo_pre": df["elo2_pre"], "win": 1 - df["home_win"],
        "margin": -df["point_margin"],
        "points_for": df["score2"], "points_against": df["score1"]})

    teams = (pd.concat([home, away])
             .groupby(["season", "team"])
             .mean()
             .reset_index()
             .rename(columns={"elo_pre": "avg_pregame_elo",
                              "win": "win_pct",
                              "margin": "avg_margin"}))
    return teams


def plot_team_season_correlation(teams: pd.DataFrame) -> None:
    """
    Figure 2: the most highly correlated meaningful pair.
    Average pregame Elo (known before games) vs. win percentage
    (the season result). These are not linked by a formula.
    """
    columns = ["avg_pregame_elo", "win_pct", "avg_margin",
               "points_for", "points_against"]
    corr = teams[columns].corr().round(2)
    print("\nTeam-season correlation matrix:")
    print(corr.to_string())

    r = corr.loc["avg_pregame_elo", "win_pct"]
    slope, intercept = np.polyfit(teams["avg_pregame_elo"],
                                  teams["win_pct"], 1)
    x_line = np.linspace(teams["avg_pregame_elo"].min(),
                         teams["avg_pregame_elo"].max(), 100)

    plt.figure(figsize=(8, 5.5))
    plt.scatter(teams["avg_pregame_elo"], teams["win_pct"],
                color="steelblue", alpha=0.7, s=25,
                label="One team-season")
    plt.plot(x_line, slope * x_line + intercept, color="darkred",
             linewidth=1.5, label="Linear fit")
    plt.xlabel("Average pregame Elo rating")
    plt.ylabel("Regular-season win percentage")
    plt.title(f"Team Strength vs. Season Win Percentage (r = {r:.2f})")
    plt.legend(loc="upper left")
    save("fig2_team_elo_vs_win_pct.png")


def plot_prob_over_score_ratio(df: pd.DataFrame) -> None:
    """Figure 3: elo_prob1 over score1/score2."""
    r = df["elo_prob1"].corr(df["score_ratio"])

    bins = np.arange(0.70, 1.41, 0.05)
    df = df.assign(ratio_bin=pd.cut(df["score_ratio"], bins))
    binned = df.groupby("ratio_bin", observed=True).agg(
        ratio=("score_ratio", "mean"), prob=("elo_prob1", "mean"))

    plt.figure(figsize=(8, 5.5))
    plt.hexbin(df["score_ratio"], df["elo_prob1"], gridsize=45,
               cmap="Blues", mincnt=1)
    plt.colorbar(label="Number of games")
    plt.plot(binned["ratio"], binned["prob"], color="black",
             marker="o", linewidth=1.5, label="Mean elo_prob1 per bin")
    plt.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xlim(0.65, 1.45)
    plt.xlabel("score1 / score2  (home points / away points)")
    plt.ylabel("elo_prob1  (pregame home win probability)")
    plt.title(f"Distribution of elo_prob1 over score1/score2 (r = {r:.2f})")
    plt.legend(loc="upper left")
    save("fig3_elo_prob1_vs_score_ratio.png")


def plot_3d_prob_over_scores(df: pd.DataFrame) -> None:
    """
    Figure 4: 3D view of elo_prob1 over score1 and score2.
    Games are grouped into 5-point score cells; each point is the
    mean elo_prob1 of a cell with at least 20 games.
    """
    cells = df.assign(
        s1=(df["score1"] // 5) * 5 + 2.5,
        s2=(df["score2"] // 5) * 5 + 2.5,
    ).groupby(["s1", "s2"]).agg(
        prob=("elo_prob1", "mean"), games=("elo_prob1", "size")
    ).reset_index()
    cells = cells[cells["games"] >= 20]

    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    points = ax.scatter(cells["s2"], cells["s1"], cells["prob"],
                        c=cells["prob"], cmap="coolwarm",
                        s=cells["games"] / 4, depthshade=False)
    fig.colorbar(points, ax=ax, shrink=0.6, pad=0.1,
                 label="Mean elo_prob1")
    ax.set_xlabel("score2 (away points)")
    ax.set_ylabel("score1 (home points)")
    ax.set_zlabel("Mean elo_prob1")
    ax.set_title("elo_prob1 over score1 and score2")
    ax.view_init(elev=22, azim=-60)
    save("fig4_3d_elo_prob1_over_scores.png")


def plot_scores_by_probability(df: pd.DataFrame) -> None:
    """Figure 5: score1 vs. score2, colored by elo_prob1."""
    r = df["score1"].corr(df["score2"])
    plt.figure(figsize=(7.5, 6.5))
    points = plt.scatter(df["score2"], df["score1"], c=df["elo_prob1"],
                         cmap="coolwarm", s=6, alpha=0.6)
    plt.colorbar(points, label="elo_prob1")
    low = min(df["score1"].min(), df["score2"].min())
    high = max(df["score1"].max(), df["score2"].max())
    plt.plot([low, high], [low, high], color="black", linewidth=1,
             label="score1 = score2")
    plt.xlabel("score2  (away points)")
    plt.ylabel("score1  (home points)")
    plt.title(f"Home vs. Away Points, Colored by elo_prob1 (r = {r:.2f})")
    plt.legend(loc="upper left")
    save("fig5_scores_colored_by_elo_prob1.png")


def plot_distributions(df: pd.DataFrame) -> None:
    """
    Figure 6: single-column distributions.
    Left: elo_diff is symmetric around zero because every pair of
    teams plays at each other's site, so each matchup appears with
    both signs.
    Right: the home point margin is shifted above zero, which shows
    the home-court advantage.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    sns.histplot(df["elo_diff"], bins=40, kde=True,
                 color="gray", ax=axes[0])
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"elo_diff (mean = {df['elo_diff'].mean():+.1f})")
    axes[0].set_xlabel("elo_diff  (elo1_pre - elo2_pre)")
    axes[0].set_ylabel("Number of games")

    mean_margin = df["point_margin"].mean()
    sns.histplot(df["point_margin"], binwidth=2, kde=True,
                 color="steelblue", ax=axes[1])
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].axvline(mean_margin, color="darkred", linewidth=1.5,
                    label=f"Mean = {mean_margin:+.1f} points")
    axes[1].set_title("Home point margin")
    axes[1].set_xlabel("Point margin  (score1 - score2)")
    axes[1].set_ylabel("Number of games")
    axes[1].legend()
    save("fig6_distributions.png")


# ----------------------------------------------------------------------
# 5. Prediction performance report
# ----------------------------------------------------------------------
def prediction_report(df: pd.DataFrame) -> None:
    """
    Logistic regression on pregame features only.
    Time-based split: train on 2014-2021, validate on 2022-2023.
    """
    features = ["elo_diff", "form1", "form2"]
    train = df[df["season"] <= TRAIN_LAST_SEASON]
    valid = df[df["season"] > TRAIN_LAST_SEASON]

    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(train[features], train["home_win"])

    results = {}
    print("\nPrediction performance report")
    print(f"Features: {features}")
    for name, part in [("Training", train), ("Validation", valid)]:
        predicted = model.predict(part[features])
        acc = accuracy_score(part["home_win"], predicted)
        cm = confusion_matrix(part["home_win"], predicted, labels=[0, 1])
        results[name] = (acc, cm)
        seasons = f"{part['season'].min()}-{part['season'].max()}"
        print(f"\n{name} ({seasons}, {len(part):,} games)")
        print(f"  Accuracy: {acc:.3f}")
        print("  Confusion matrix (rows = actual, cols = predicted)")
        print("                 Pred away  Pred home")
        print(f"  Actual away    {cm[0, 0]:>9,}  {cm[0, 1]:>9,}")
        print(f"  Actual home    {cm[1, 0]:>9,}  {cm[1, 1]:>9,}")

    # Simple baselines on the validation seasons for context.
    home_rule = valid["home_win"].mean()
    elo_rule = ((valid["elo_prob1"] > 0.5).astype(int)
                == valid["home_win"]).mean()
    print(f"\nValidation baselines: always pick home = {home_rule:.3f}, "
          f"pick higher elo_prob1 = {elo_rule:.3f}")

    # Figure 7: confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, (name, (acc, cm)) in zip(axes, results.items()):
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
                    ax=ax, annot_kws={"size": 12},
                    xticklabels=["Away win", "Home win"],
                    yticklabels=["Away win", "Home win"])
        ax.set_title(f"{name} set - accuracy {acc:.1%}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    save("fig7_confusion_matrices.png")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    raw = load_raw_data(DATA_PATH)
    raw = add_recent_form(raw)
    df = raw[raw["season"].between(MIN_SEASON, MAX_SEASON)].copy()
    print(f"Games {MIN_SEASON}-{MAX_SEASON}: {len(df):,}")
    print(f"Home win rate: {df['home_win'].mean():.3f}")

    check_elo_formula(df)
    plot_correlation_matrix(df)
    plot_team_season_correlation(build_team_seasons(df))
    plot_prob_over_score_ratio(df)
    plot_3d_prob_over_scores(df)
    plot_scores_by_probability(df)
    plot_distributions(df)
    prediction_report(df)


if __name__ == "__main__":
    main()
