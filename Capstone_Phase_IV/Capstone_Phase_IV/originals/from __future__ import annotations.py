from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

from sklearn.base import clone
from sklearn.linear_model import ElasticNetCV, LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Optional XGBoost support
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

RAW_URL = "https://r2.datahub.io/cm2uksgvo0000la0c6y682zw2/main/raw/datasets/nba-carmelo/data/nba_elo.csv"
RAW_PATH = BASE_DIR / "basketball_games_raw_538.csv"
DATA_PATH = BASE_DIR / "basketball_games.csv"

SEASON_COL = "season"
DATE_COL = "game_date"
TARGET_COL = "target"

CORRELATION_THRESHOLD = 0.90
RANDOM_STATE = 100

ELASTICNET_ALPHAS = np.logspace(-4, 1, 50)
ELASTICNET_L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 1.0]


@dataclass
class FoldResult:
    season_tested: str
    model_name: str
    n_features: int
    accuracy: float
    precision: float
    recall: float
    f1: float


# -----------------------------
# Download + prepare raw data
# -----------------------------
def download_raw_csv(url: str, dest: Path) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,text/plain,application/octet-stream,*/*",
    }

    print("Downloading raw FiveThirtyEight data...")
    response = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    response.raise_for_status()

    prefix = response.content[:512].lstrip().lower()
    if prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html"):
        raise RuntimeError(
            "Downloaded HTML instead of CSV.\n"
            f"Final URL: {response.url}\n"
            f"Content-Type: {response.headers.get('Content-Type')}"
        )

    dest.write_bytes(response.content)
    print(f"Saved raw file: {dest}")


def validate_raw_csv(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}")

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()

    if not lines:
        raise RuntimeError("Raw CSV is empty.")

    first_line = lines[0].strip().lower()
    if not first_line.startswith("date,season,neutral,playoff,team1,team2"):
        raise RuntimeError(
            "The saved file does not look like the expected nba_elo.csv file.\n"
            f"First line was:\n{lines[0]}"
        )

    print("Raw CSV validated successfully.")
    print("First 3 lines:")
    for line in lines[:3]:
        print(line)


def prepare_basketball_games(raw_path: Path, out_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path, encoding="utf-8-sig", low_memory=False)

    required = ["date", "season", "elo1_pre", "elo2_pre", "elo_prob1", "score1", "score2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw CSV: {missing}")

    use = df.dropna(subset=["score1", "score2"]).copy()

    use["game_date"] = pd.to_datetime(use["date"], errors="coerce")
    if use["game_date"].isna().any():
        bad = int(use["game_date"].isna().sum())
        raise ValueError(f"Found {bad} invalid dates in raw CSV.")

    use["season"] = pd.to_numeric(use["season"], errors="raise").astype(int)
    use["elo1_pre"] = pd.to_numeric(use["elo1_pre"], errors="raise")
    use["elo2_pre"] = pd.to_numeric(use["elo2_pre"], errors="raise")
    use["elo_prob1"] = pd.to_numeric(use["elo_prob1"], errors="raise")
    use["score1"] = pd.to_numeric(use["score1"], errors="raise")
    use["score2"] = pd.to_numeric(use["score2"], errors="raise")

    # Build the exact format your ML code needs
    out = pd.DataFrame({
        "season": use["season"],
        "game_date": use["game_date"].dt.strftime("%Y-%m-%d"),
        "target": (use["score1"] > use["score2"]).astype(int),
        "elo1_pre": use["elo1_pre"],
        "elo2_pre": use["elo2_pre"],
        "elo_prob1": use["elo_prob1"],
    })

    # Extra pregame feature
    out["elo_diff"] = out["elo1_pre"] - out["elo2_pre"]

    out = out.dropna().sort_values(["game_date", "season"]).reset_index(drop=True)
    out.to_csv(out_path, index=False)

    print(f"Saved prepared dataset: {out_path}")
    print("\nPrepared dataset preview:")
    print(out.head(10).to_string(index=False))

    return out


# -----------------------------
# Data loading / validation
# -----------------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {SEASON_COL, DATE_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if df[DATE_COL].isna().any():
        raise ValueError(f"Column '{DATE_COL}' contains invalid dates.")

    df = df.sort_values([DATE_COL, SEASON_COL]).reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {SEASON_COL, DATE_COL, TARGET_COL}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in excluded]

    if not feature_cols:
        raise ValueError("No numeric feature columns found.")
    return feature_cols


# -----------------------------
# Feature selection helpers
# -----------------------------
def run_elasticnet_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
) -> List[str]:
    n_splits = min(5, max(2, len(X) // 50))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    selector = ElasticNetCV(
        alphas=ELASTICNET_ALPHAS,
        l1_ratio=ELASTICNET_L1_RATIOS,
        cv=tscv,
        max_iter=10000,
        random_state=RANDOM_STATE,
    )
    selector.fit(X, y)

    coefs = pd.Series(selector.coef_, index=feature_names)
    selected = coefs[coefs.abs() > 1e-12].index.tolist()

    if not selected:
        selected = feature_names.copy()

    return selected


def remove_high_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.90,
) -> List[str]:
    corr_matrix = X.corr().abs()
    target_corr = X.apply(lambda col: col.corr(y)).abs().fillna(0.0)

    cols = list(X.columns)
    to_drop = set()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            if c1 in to_drop or c2 in to_drop:
                continue

            pair_corr = corr_matrix.loc[c1, c2]
            if pd.notna(pair_corr) and pair_corr > threshold:
                if target_corr[c1] >= target_corr[c2]:
                    to_drop.add(c2)
                else:
                    to_drop.add(c1)

    kept = [c for c in cols if c not in to_drop]
    return kept


def select_features(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
) -> List[str]:
    selected = run_elasticnet_feature_selection(
        X_train_scaled, y_train, X_train_scaled.columns.tolist()
    )
    X_selected = X_train_scaled[selected]
    final_features = remove_high_correlation(
        X_selected, y_train, threshold=CORRELATION_THRESHOLD
    )
    return final_features


# -----------------------------
# Models
# -----------------------------
def get_models() -> Dict[str, object]:
    models = {
        "LR": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "RR": RidgeClassifier(random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "NB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=9),
        "SVM": SVC(probability=False, random_state=RANDOM_STATE),
        "BAG": BaggingClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=RANDOM_STATE,
            early_stopping=True,
        ),
        "AB": AdaBoostClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }

    if XGB_AVAILABLE:
        models["XGB"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )

    stack_estimators = [
        ("lr", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ("svm", SVC(kernel="rbf", probability=False, random_state=RANDOM_STATE)),
    ]
    models["STACK"] = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        n_jobs=-1,
    )

    return models


# -----------------------------
# Season-based training/testing
# -----------------------------
def season_walk_forward_split(df: pd.DataFrame) -> List[Tuple[List[str], str]]:
    seasons = sorted(df[SEASON_COL].astype(str).unique().tolist())
    if len(seasons) < 3:
        raise ValueError("Need at least 3 seasons for walk-forward evaluation.")

    splits = []
    for i in range(2, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]
        splits.append((train_seasons, test_season))
    return splits


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_experiment(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = get_feature_columns(df)
    models = get_models()
    splits = season_walk_forward_split(df)

    all_results: List[FoldResult] = []

    for train_seasons, test_season in splits:
        train_df = df[df[SEASON_COL].astype(str).isin(train_seasons)].copy()
        test_df = df[df[SEASON_COL].astype(str) == str(test_season)].copy()

        X_train = train_df[feature_cols].copy()
        y_train = train_df[TARGET_COL].astype(int).copy()

        X_test = test_df[feature_cols].copy()
        y_test = test_df[TARGET_COL].astype(int).copy()

        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feature_cols,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=feature_cols,
            index=X_test.index,
        )

        final_features = select_features(X_train_scaled, y_train)

        X_train_final = X_train_scaled[final_features]
        X_test_final = X_test_scaled[final_features]

        print(f"\n=== Test season: {test_season} ===")
        print(f"Train seasons: {train_seasons}")
        print(f"Selected features: {len(final_features)}")
        print(f"Features used: {final_features}")

        for model_name, model in models.items():
            clf = clone(model)
            clf.fit(X_train_final, y_train)
            y_pred = clf.predict(X_test_final)

            metrics = evaluate_predictions(y_test, y_pred)

            result = FoldResult(
                season_tested=str(test_season),
                model_name=model_name,
                n_features=len(final_features),
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
            )
            all_results.append(result)

            print(
                f"{model_name:>5} | "
                f"Acc: {metrics['accuracy']:.4f} | "
                f"Prec: {metrics['precision']:.4f} | "
                f"Rec: {metrics['recall']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )

    return pd.DataFrame([r.__dict__ for r in all_results])


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    if not RAW_PATH.exists():
        download_raw_csv(RAW_URL, RAW_PATH)
    else:
        print(f"Using existing raw file: {RAW_PATH}")

    validate_raw_csv(RAW_PATH)
    prepare_basketball_games(RAW_PATH, DATA_PATH)

    df = load_data(DATA_PATH)
    results_df = run_experiment(df)

    print("\n=== Average Results by Model ===")
    summary = (
        results_df
        .groupby("model_name")[["accuracy", "precision", "recall", "f1"]]
        .mean()
        .sort_values("accuracy", ascending=False)
        .round(4)
    )
    print(summary)

    results_df.to_csv(BASE_DIR / "season_model_results.csv", index=False)
    summary.to_csv(BASE_DIR / "model_summary.csv")

    print("\nSaved files:")
    print(f"- {RAW_PATH}")
    print(f"- {DATA_PATH}")
    print(f"- {BASE_DIR / 'season_model_results.csv'}")
    print(f"- {BASE_DIR / 'model_summary.csv'}")


if __name__ == "__main__":
    main()