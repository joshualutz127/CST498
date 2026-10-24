"""Offline NBA integration runner. See README.md for the demonstration sequence."""
from pathlib import Path
import argparse
import json
import platform
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
import nba_visualization as viz

BASE = Path(__file__).resolve().parent
FEATURES = ['elo_diff', 'form1', 'form2']


def prepare_data(path):
    """Reuse the latest loader and past-game form calculation, then validate."""
    df = viz.load_raw_data(Path(path))
    # Validate the selected evaluation era; early archive forfeits are out of scope.
    df = df[df.season.between(2014, 2023)].copy().reset_index(drop=True)
    # Historical records can share a date and matchup with different scores.
    # Reject exact repeated records rather than assuming that key is unique.
    if df.duplicated().any():
        raise ValueError('Duplicate game identifiers found.')
    numeric = ['season', 'elo1_pre', 'elo2_pre', 'elo_prob1', 'score1', 'score2']
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite numeric values found.')
    if (df[['score1', 'score2']] <= 0).any().any():
        raise ValueError('Final scores must be positive.')
    if (df.score1 == df.score2).any():
        raise ValueError('Tied final scores cannot define a basketball winner.')
    if not df.elo_prob1.between(0, 1).all():
        raise ValueError('Elo probabilities must be between zero and one.')
    df = viz.add_recent_form(df)
    df = df[df.season.between(2014, 2023)].copy()
    if not np.isfinite(df[FEATURES].to_numpy()).all():
        raise ValueError('Invalid model features.')
    return df


def split_data(df):
    """Freeze training, validation and final test by season, without shuffling."""
    parts = {'Training': df[df.season <= 2021],
             'Validation': df[df.season == 2022], 'Test': df[df.season == 2023]}
    if any(p.empty for p in parts.values()):
        raise ValueError('Training 2014-2021, validation 2022 and test 2023 must exist.')
    if not (parts['Training'].date.max() < parts['Validation'].date.min()
            and parts['Validation'].date.max() < parts['Test'].date.min()):
        raise ValueError('Chronological partitions overlap.')
    return parts


def run(data_path, output):
    """Train both fixed models on identical features and export reproducible evidence."""
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df = prepare_data(data_path)
    parts = split_data(df)
    models = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, random_state=100)),
        'Random Forest': RandomForestClassifier(n_estimators=300,
            random_state=100, n_jobs=1),
    }
    rows, predictions = [], []
    print('NBA integration and testing run')
    print('Features:', FEATURES)
    print('Fixed models; no validation or test tuning; threshold 0.5')
    for name, part in parts.items():
        print(f'{name}: {len(part)} games, {part.date.min().date()} to {part.date.max().date()}')
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for row, (name, model) in enumerate(models.items()):
        fit_start = time.perf_counter()
        model.fit(parts['Training'][FEATURES], parts['Training'].home_win)
        fit_seconds = time.perf_counter() - fit_start
        for col, (partition, part) in enumerate(parts.items()):
            prob = model.predict_proba(part[FEATURES])[:, 1]
            predicted = (prob >= .5).astype(int)
            actual = part.home_win.to_numpy()
            cm = confusion_matrix(actual, predicted, labels=[0, 1])
            result = dict(model=name, partition=partition, games=len(part),
                accuracy=float(accuracy_score(actual, predicted)),
                log_loss=float(log_loss(actual, prob, labels=[0, 1])),
                fit_seconds=fit_seconds, tn=int(cm[0, 0]), fp=int(cm[0, 1]),
                fn=int(cm[1, 0]), tp=int(cm[1, 1]))
            rows.append(result)
            exported = part[['date', 'season', 'team1', 'team2', 'home_win']].copy()
            exported['model'], exported['partition'] = name, partition
            exported['probability'], exported['prediction'] = prob, predicted
            predictions.append(exported)
            print(f'{name} | {partition} | accuracy {result["accuracy"]:.4f} | log loss {result["log_loss"]:.4f}')
            ConfusionMatrixDisplay(cm, display_labels=['Team 2', 'Team 1']).plot(
                ax=axes[row, col], colorbar=False, cmap='Blues')
            axes[row, col].set_title(f'{name}\n{partition}: {result["accuracy"]:.1%}')
    fig.tight_layout()
    fig.savefig(output / 'phase4_confusion_matrices.png', dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output / 'model_metrics.csv', index=False)
    pd.concat(predictions).to_csv(output / 'predictions.csv', index=False)
    # Baselines use training class frequency for the constant probability model.
    train_rate = parts['Training'].home_win.mean()
    baseline_rows = []
    for name, part in parts.items():
        for rule, prob in [('Training class frequency', np.full(len(part), train_rate)),
                           ('Supplied Elo probability', part.elo_prob1.to_numpy())]:
            baseline_rows.append(dict(partition=name, baseline=rule,
                accuracy=accuracy_score(part.home_win, prob >= .5),
                log_loss=log_loss(part.home_win, prob, labels=[0, 1])))
    pd.DataFrame(baseline_rows).to_csv(output / 'baselines.csv', index=False)
    metadata = dict(features=FEATURES, data_file=Path(data_path).name,
        python=platform.python_version(), pandas=pd.__version__,
        numpy=np.__version__, sklearn=sklearn.__version__,
        total_seconds=time.perf_counter()-started,
        split_counts={name: len(part) for name, part in parts.items()},
        evaluation='Sequential historical backtest; previous completed games update recent form.')
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    print('Evidence saved to', output)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=BASE/'data'/'basketball_games_raw_538.csv')
    parser.add_argument('--output', type=Path, default=BASE/'results')
    args = parser.parse_args()
    run(args.data, args.output)
