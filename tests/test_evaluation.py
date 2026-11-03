"""Synthetic score-ledger tests; these fixtures are not actual match records."""

import math
import os
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluate import report_staging, run_evaluation
from src.evaluation import (ModelSpec, calibration_bins, evaluate_predictions, paired_differences,
                            probability_metrics, validate_predictions, week_blocks)


def fixture():
    dates = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-11", "2021-01-12",
                            "2021-01-18", "2021-01-19", "2021-01-25", "2021-01-26"], utc=True)
    matches = pd.DataFrame({"match_id": [f"synthetic_{i}" for i in range(8)], "period": "development_test",
                            "date": dates, "league": ["WLDs", "MSI", "WLDs", "Example"] * 2,
                            "team_a": "Synthetic A", "team_b": "Synthetic B", "result_a": [0, 1] * 4,
                            "international_scope": [True, True, True, False] * 2})
    models = [ModelSpec("global_logistic_v1", "Synthetic baseline", "1"),
              ModelSpec("candidate_v2", "Synthetic candidate", "2", fallback_model_id="global_logistic_v1")]
    base = np.array([.4, .6] * 4)
    candidate = np.array([.2, .8] * 4);candidate[-1] = base[-1]
    ledger = []
    for model, probabilities in zip(models, [base, candidate]):
        available = np.ones(8, dtype=bool)
        if model.model_id == "candidate_v2":available[-1] = False
        ledger.append(pd.DataFrame({"match_id": matches.match_id, "period": matches.period,
                                    "model_id": model.model_id, "probability_a": probabilities,
                                    "prediction_timestamp": dates, "information_cutoff_exclusive": dates,
                                    "model_trained_through": "2020-12-31T00:00:00Z", "configuration_selected_through": None,
                                    "native_available": available, "used_fallback": ~available,
                                    "parameter_selection_reused": False, "evaluation_mode": "online_elo"}))
    return matches, pd.concat(ledger, ignore_index=True), models


class EvaluationTests(unittest.TestCase):
    def test_probability_metrics_and_50_percent_tie_rule(self):
        scores = probability_metrics([0, 1], [.5, .5])
        self.assertAlmostEqual(scores['log_loss'], math.log(2))
        self.assertEqual(scores['brier'], .25)
        self.assertEqual(scores['accuracy'], .5)
        self.assertEqual(scores['roc_auc'], .5)
        self.assertEqual(probability_metrics([1, 1], [.5, .5])['accuracy'], 1)
        self.assertTrue(math.isnan(probability_metrics([1, 1], [.5, .5])['roc_auc']))

    def test_exact_zero_one_and_empty_scores(self):
        scores = probability_metrics([0, 1], [0, 1])
        self.assertEqual(scores['accuracy'], 1)
        self.assertEqual(scores['brier'], 0)
        self.assertTrue(math.isfinite(scores['log_loss']))
        self.assertEqual(probability_metrics([], [])['matches'], 0)
        for y, p in [([.5], [.5]), ([1], [float('nan')]), ([1], [2]), ([0, 1], [.5])]:
            with self.assertRaises(ValueError):probability_metrics(y, p)

    def test_calibration_boundaries_and_counts(self):
        bins = calibration_bins([0, 1, 0, 1], [0, .1, .5, 1])
        self.assertEqual(bins.matches.sum(), 4)
        self.assertEqual(bins.loc[0, 'matches'], 2)
        self.assertEqual(bins.loc[4, 'matches'], 1)
        self.assertEqual(bins.loc[9, 'matches'], 1)
        self.assertTrue(bins.loc[bins.matches.eq(0), 'observed_win_rate'].isna().all())

    def test_full_and_common_population_counts(self):
        matches, ledger, models = fixture()
        tables = evaluate_predictions(matches, ledger, models, resamples=100)
        report = tables['metrics'].loc[tables['metrics'].scope.eq('all')]
        self.assertTrue(report.loc[report.population.eq('full_scope'), 'matches'].eq(8).all())
        self.assertTrue(report.loc[report.population.eq('common_native'), 'matches'].eq(7).all())
        coverage = tables['coverage'].loc[tables['coverage'].scope.eq('all') & tables['coverage'].population.eq('full_scope')].set_index('model_id')
        self.assertEqual(coverage.loc['candidate_v2', 'fallback_predictions'], 1)
        self.assertEqual(coverage.loc['global_logistic_v1', 'fallback_predictions'], 0)
        self.assertTrue(tables['calibration'].groupby(['period', 'scope', 'population', 'model_id']).matches.sum().gt(0).all())

    def test_population_mismatch_duplicates_and_unknown_models(self):
        matches, ledger, models = fixture()
        bad_cases = [ledger.iloc[:-1], pd.concat([ledger, ledger.iloc[:1]], ignore_index=True)]
        changed = ledger.copy();changed.loc[0, 'model_id'] = 'unregistered';bad_cases.append(changed)
        for bad in bad_cases:
            with self.assertRaises(ValueError):validate_predictions(matches, bad, models)

    def test_missing_predictions_and_invalid_flags_fail(self):
        matches, ledger, models = fixture()
        for column, value in [('probability_a', np.nan), ('probability_a', 1.1), ('native_available', 'true')]:
            bad = ledger.copy()
            if column == 'native_available':bad[column] = bad[column].astype(object)
            bad.loc[0, column] = value
            with self.assertRaises(ValueError):validate_predictions(matches, bad, models)

    def test_training_prediction_and_information_cutoffs(self):
        matches, ledger, models = fixture()
        for column, value in [('model_trained_through', '2022-01-01T00:00:00Z'),
                               ('prediction_timestamp', pd.Timestamp('2022-01-01', tz='UTC')),
                               ('information_cutoff_exclusive', pd.Timestamp('2022-01-01', tz='UTC'))]:
            bad = ledger.copy();bad.loc[0, column] = value
            with self.assertRaises(ValueError):validate_predictions(matches, bad, models)
        bad = ledger.copy();bad['prediction_timestamp'] = '2021-01-04'
        with self.assertRaises(ValueError):validate_predictions(matches, bad, models)

    def test_selection_reuse_requires_explicit_validation_label(self):
        matches, ledger, models = fixture()
        models[1] = replace(models[1], selection_period='validation')
        candidate = ledger.model_id.eq('candidate_v2')
        ledger.loc[candidate, 'configuration_selected_through'] = '2022-01-01T00:00:00Z'
        with self.assertRaises(ValueError):validate_predictions(matches, ledger, models)
        ledger.loc[candidate, 'parameter_selection_reused'] = True
        with self.assertRaises(ValueError):validate_predictions(matches, ledger, models)
        matches['period'] = 'validation';ledger['period'] = 'validation'
        validate_predictions(matches, ledger, models)

    def test_mixed_modes_rejected(self):
        matches, ledger, models = fixture()
        ledger.loc[0, 'evaluation_mode'] = 'frozen_pre_tournament'
        with self.assertRaises(ValueError):validate_predictions(matches, ledger, models)

    def test_same_forecast_timing_required_even_in_frozen_mode(self):
        matches, ledger, models = fixture()
        ledger['evaluation_mode'] = 'frozen_pre_tournament'
        ledger['information_cutoff_exclusive'] = pd.Timestamp('2021-01-01', tz='UTC')
        ledger['prediction_timestamp'] = pd.Timestamp('2021-01-01', tz='UTC')
        validate_predictions(matches, ledger, models, 'frozen_pre_tournament')
        ledger.loc[0, 'information_cutoff_exclusive'] = pd.Timestamp('2020-12-31T12:00:00Z')
        with self.assertRaises(ValueError):validate_predictions(matches, ledger, models, 'frozen_pre_tournament')

    def test_equal_timestamp_cannot_cross_periods(self):
        matches, ledger, models = fixture()
        matches.loc[1, 'date'] = matches.loc[0, 'date']
        matches.loc[1, 'period'] = 'validation'
        ledger.loc[ledger.match_id.eq('synthetic_1'), 'period'] = 'validation'
        with self.assertRaises(ValueError):validate_predictions(matches, ledger, models)

    def test_prediction_specific_outcomes_are_rejected(self):
        matches, ledger, models = fixture()
        ledger['result_a'] = 1
        with self.assertRaises(ValueError):validate_predictions(matches, ledger, models)

    def test_fallback_matches_registered_reference(self):
        matches, ledger, models = fixture()
        ledger.loc[ledger.used_fallback, 'probability_a'] = .9
        with self.assertRaises(ValueError):validate_predictions(matches, ledger, models)

    def test_bootstrap_determinism_and_identical_predictions(self):
        blocks = ['a', 'a', 'b', 'b']
        first = paired_differences([0, 1, 0, 1], [.2, .8, .2, .8], [.4, .6, .4, .6], blocks, resamples=100)
        second = paired_differences([0, 1, 0, 1], [.2, .8, .2, .8], [.4, .6, .4, .6], blocks, resamples=100)
        pd.testing.assert_frame_equal(first, second)
        same = paired_differences([0, 1, 0, 1], [.4, .6, .4, .6], [.4, .6, .4, .6], blocks, resamples=100)
        self.assertTrue(same[['candidate_minus_baseline', 'ci_lower', 'ci_upper']].eq(0).all().all())

    def test_week_blocks_use_utc_monday(self):
        dates = pd.Series(pd.to_datetime(['2021-01-03T23:59:00Z', '2021-01-04T00:00:00Z', '2021-01-10T23:59:00Z']))
        blocks = week_blocks(dates)
        self.assertNotEqual(blocks.iloc[0], blocks.iloc[1])
        self.assertEqual(blocks.iloc[1], blocks.iloc[2])

    def test_game_weighted_not_unweighted_week_means(self):
        result = paired_differences([1]*10, [.9]+[.1]*9, [.1]*10, ['small']+['large']*9, resamples=100)
        self.assertAlmostEqual(result.set_index('metric').loc['accuracy', 'candidate_minus_baseline'], .1)

    def test_insufficient_blocks_have_no_interval(self):
        result = paired_differences([0, 1], [.2, .8], [.4, .6], ['one', 'one'], resamples=100)
        self.assertTrue(result.ci_lower.isna().all())
        self.assertTrue(result.interval_status.eq('insufficient_blocks').all())

    def test_shuffle_and_nonmutation(self):
        matches, ledger, models = fixture()
        matches_before, ledger_before = matches.copy(deep=True), ledger.copy(deep=True)
        first = evaluate_predictions(matches, ledger, models, resamples=100)
        second = evaluate_predictions(matches.sample(frac=1, random_state=1), ledger.sample(frac=1, random_state=2), models, resamples=100)
        for key in first:pd.testing.assert_frame_equal(first[key], second[key])
        pd.testing.assert_frame_equal(matches, matches_before);pd.testing.assert_frame_equal(ledger, ledger_before)

    def test_no_native_rows_are_reported_as_empty_not_dropped(self):
        matches, ledger, models = fixture()
        candidate = ledger.model_id.eq('candidate_v2')
        ledger.loc[candidate, 'native_available'] = False
        ledger.loc[candidate, 'used_fallback'] = True
        ledger.loc[candidate, 'probability_a'] = [.4, .6]*4
        result = evaluate_predictions(matches, ledger, models, resamples=100)
        self.assertTrue(result['metrics'].loc[result['metrics'].population.eq('common_native'), 'matches'].eq(0).all())

    def test_invalid_run_settings(self):
        matches, ledger, models = fixture()
        for options in ({'resamples':0}, {'seed':-1}, {'bins':0}, {'confidence':1}, {'baseline_model_id':'missing'}):
            with self.assertRaises(ValueError):evaluate_predictions(matches, ledger, models, **options)


class VersionedRunTests(unittest.TestCase):
    def test_staging_cleanup_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = None
            with self.assertRaisesRegex(ValueError, 'synthetic failure'):
                with report_staging(directory) as staging:
                    self.assertEqual(staging.parent, Path(directory).resolve())
                    (staging/'partial.txt').write_text('synthetic test', encoding='utf-8')
                    raise ValueError('synthetic failure')
            self.assertFalse(staging.exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows ACL regression')
    def test_staging_inherits_workspace_permissions_on_windows(self):
        # Use the normal checkout folder, not a deliberately private tempfile.
        parent = Path(__file__).resolve().parent
        powershell = shutil.which('pwsh.exe') or shutil.which('powershell.exe')
        if powershell is None:
            self.skipTest('PowerShell is unavailable for the Windows ACL check.')
        def protected(path):
            literal = str(path).replace("'", "''")
            result = subprocess.check_output(
                [powershell, '-NoProfile', '-Command', f"(Get-Acl -LiteralPath '{literal}').AreAccessRulesProtected"],
                text=True,
            ).strip()
            return result == 'True'
        if protected(parent):
            self.skipTest('Checkout deliberately has protected ACLs.')
        with report_staging(parent) as staging:
            self.assertFalse(protected(staging), 'Published outputs must inherit normal project ACLs.')
            output = staging/'synthetic-run'
            output.mkdir()
            self.assertFalse(protected(output))

    def test_permission_error_is_not_reported_as_incomplete(self):
        matches, ledger, models = fixture()
        source = {'input_sha256': {'synthetic_fixture': 'fixture-only'}, 'periods': {}, 'selection_end': '2020-12-31T00:00:00Z'}
        with tempfile.TemporaryDirectory() as directory:
            with patch('src.evaluate.load_saved_suite', return_value=(matches, ledger, models, source)), patch('src.evaluate.verify_existing_scores'):
                run_evaluation(directory, resamples=100)
                original = Path.read_text
                def cannot_read(path, *args, **kwargs):
                    if path.name == 'manifest.json':
                        raise PermissionError('synthetic access denial')
                    return original(path, *args, **kwargs)
                with patch.object(Path, 'read_text', cannot_read):
                    with self.assertRaisesRegex(ValueError, 'Check report-folder permissions'):
                        run_evaluation(directory, resamples=100)

    def test_idempotent_immutable_run_and_changed_configuration(self):
        matches, ledger, models = fixture()
        source = {'input_sha256': {'synthetic_fixture': 'fixture-only'}, 'periods': {}, 'selection_end': '2020-12-31T00:00:00Z'}
        with tempfile.TemporaryDirectory() as directory:
            with patch('src.evaluate.load_saved_suite', return_value=(matches, ledger, models, source)), patch('src.evaluate.verify_existing_scores'):
                first = run_evaluation(directory, resamples=100)
                before = (first/'manifest.json').read_bytes()
                second = run_evaluation(directory, resamples=100)
                self.assertEqual(first, second)
                self.assertEqual(before, (second/'manifest.json').read_bytes())
                third = run_evaluation(directory, resamples=100, seed=1)
                self.assertNotEqual(first, third)
                self.assertTrue((first/'report.md').is_file())
                self.assertTrue((first/'predictions.csv.gz').is_file())
                (first/'metrics.csv').write_text('changed', encoding='utf-8')
                with self.assertRaises(ValueError):run_evaluation(directory, resamples=100)


if __name__ == '__main__':
    unittest.main()
