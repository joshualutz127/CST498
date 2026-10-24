"""Behavioral checks for the real data, chronology and invalid-input handling."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import nba_visualization as viz
from run_phase4 import BASE, FEATURES, prepare_data, split_data


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = BASE/'data'/'basketball_games_raw_538.csv'
        cls.raw = pd.read_csv(cls.path)
        cls.raw = cls.raw[cls.raw.season == 2014].reset_index(drop=True)
        cls.df = prepare_data(cls.path)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            viz.load_raw_data(BASE/'missing_input.csv')

    def test_missing_column(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'missing_column.csv'
            self.raw.head(5).drop(columns='elo_prob1').to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'elo_prob1'):
                viz.load_raw_data(path)

    def test_chronological_partitions(self):
        parts = split_data(self.df)
        self.assertEqual(sum(map(len, parts.values())), len(self.df))
        self.assertTrue(parts['Training'].date.max() < parts['Validation'].date.min())
        self.assertTrue(parts['Validation'].date.max() < parts['Test'].date.min())
        self.assertFalse(set(parts['Training'].index) & set(parts['Test'].index))

    def test_missing_test_season(self):
        with self.assertRaisesRegex(ValueError, 'must exist'):
            split_data(self.df[self.df.season != 2023])

    def test_current_score_not_used_in_own_features(self):
        raw = viz.load_raw_data(self.path).tail(200).reset_index(drop=True)
        before = viz.add_recent_form(raw)
        modified = raw.copy()
        idx = 100
        modified.loc[idx, 'point_margin'] += 1000
        after = viz.add_recent_form(modified)
        np.testing.assert_allclose(before.loc[:idx, FEATURES].to_numpy(dtype=float),
                                   after.loc[:idx, FEATURES].to_numpy(dtype=float))
        self.assertTrue((before.loc[idx+1:, ['form1', 'form2']] !=
                         after.loc[idx+1:, ['form1', 'form2']]).any().any())

    def test_duplicate_game_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'duplicate.csv'
            pd.concat([self.raw.head(5), self.raw.head(1)]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                prepare_data(path)

    def test_probability_range_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bad_probability.csv'
            bad = self.raw.head(5).copy()
            bad.loc[0, 'elo_prob1'] = 1.2
            bad.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'between zero and one'):
                prepare_data(path)

    def test_metric_exports_reconcile(self):
        from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
        metrics = pd.read_csv(BASE/'results'/'model_metrics.csv')
        predictions = pd.read_csv(BASE/'results'/'predictions.csv')
        self.assertEqual(len(metrics), 6)
        for row in metrics.itertuples():
            part = predictions[(predictions.model == row.model) &
                               (predictions.partition == row.partition)]
            self.assertEqual(len(part), row.games)
            self.assertAlmostEqual(accuracy_score(part.home_win, part.prediction), row.accuracy)
            self.assertAlmostEqual(log_loss(part.home_win, part.probability, labels=[0, 1]), row.log_loss)
            self.assertEqual(confusion_matrix(part.home_win, part.prediction, labels=[0, 1]).ravel().tolist(),
                             [row.tn, row.fp, row.fn, row.tp])


if __name__ == '__main__':
    unittest.main(verbosity=2)
