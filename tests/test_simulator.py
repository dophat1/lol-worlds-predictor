"""Synthetic mechanics tests; no fixture is an actual match or evaluation."""

import copy
import json
import math
import unittest
from datetime import datetime, timedelta, timezone

from src.prediction import Predictor, RatingState, validate_artifact
from src.tournament import add_result, exact_series_probability, new_session, replay_session, simulate


def fixture():
    teams = [f"Synthetic {i}" for i in range(8)]
    state = RatingState(global_ratings=dict.fromkeys(teams, 1500.0))
    for i, team in enumerate(teams):
        pool = f"R{i % 2}"
        state.homes[team] = pool
        state.local_ratings.setdefault(pool, {})[team] = 1500.0
    state.offsets = {"R0": 0.0, "R1": 0.0}
    artifact = {"schema_version": 1, "kind": "trained_model", "cutoff_exclusive": "2020-01-02T00:00:00Z",
                "model_trained_through": "2020-01-01T00:00:00Z", "last_history_timestamp": "2020-01-01T00:00:00Z",
                "state": state.as_dict(), "predictor": Predictor(math.log(10) / 400, (.004, .006), .8).as_dict(),
                "metadata": {"description": "Synthetic test fixture only"}}
    bracket = {"label": "Synthetic test bracket", "quarterfinals": [teams[i:i + 2] for i in range(0, 8, 2)]}
    return artifact, new_session(artifact, bracket)


def record(artifact, session, series, winner):
    time = datetime(2020, 1, 2, tzinfo=timezone.utc) + timedelta(minutes=len(session["events"]))
    return add_result(artifact, session, event_id=f"synthetic_{len(session['events'])}", series=series,
                      winner=winner, timestamp=time.isoformat())


class SimulatorTests(unittest.TestCase):
    def test_exact_series_matches_binomial_and_partial_scores(self):
        for p in (0, .2, .5, .8, 1):
            binomial = sum(math.comb(5, k) * p**k * (1-p)**(5-k) for k in range(3, 6))
            self.assertAlmostEqual(exact_series_probability(p), binomial)
            self.assertAlmostEqual(exact_series_probability(p, 2, 1), 1 - exact_series_probability(1-p, 1, 2))
        self.assertEqual(exact_series_probability(.5, 2, 1), .75)
        self.assertEqual(exact_series_probability(.3, 3, 2), 1)
        self.assertEqual(exact_series_probability(.9, 1, 3), 0)
        for args in [(.5, 3, 3), (float('nan'), 0, 0), (.5, -1, 0), (.5, True, 0)]:
            with self.assertRaises(ValueError):
                exact_series_probability(*args)

    def test_model_symmetry_and_fallback(self):
        artifact, _ = fixture()
        state = RatingState.from_dict(artifact['state'])
        state.global_ratings['Synthetic 0'] += 60
        predictor = Predictor(**artifact['predictor'])
        self.assertAlmostEqual(predictor.predict(state, 'Synthetic 0', 'Synthetic 1')['probability_a'] +
                               predictor.predict(state, 'Synthetic 1', 'Synthetic 0')['probability_a'], 1)
        del state.homes['Synthetic 1']
        prediction = predictor.predict(state, 'Synthetic 0', 'Synthetic 1')
        self.assertTrue(prediction['fallback'])
        self.assertEqual(prediction['probability_a'], prediction['global_probability_a'])

    def test_actual_updates_freeze_timestamp_and_leave_local_elo_unchanged(self):
        artifact, _ = fixture()
        state = RatingState.from_dict(artifact['state'])
        local_before = copy.deepcopy(state.local_ratings)
        games = [dict(team_a='Synthetic 0', team_b='Synthetic 1', result_a=1, use_region_calibration=True),
                 dict(team_a='Synthetic 0', team_b='Synthetic 3', result_a=0, use_region_calibration=True)]
        state.apply_batch(games)
        self.assertEqual(state.global_ratings['Synthetic 0'], 1500)
        self.assertEqual(state.global_ratings['Synthetic 1'], 1488)
        self.assertEqual(state.local_ratings, local_before)
        self.assertEqual(state.offsets, {'R0': 0.0, 'R1': 0.0})
        self.assertEqual(state.calibration_games, {'R0': 2, 'R1': 2})
        self.assertAlmostEqual(sum(state.global_ratings.values()), 8*1500)

    def test_unknown_and_same_pool_do_not_calibrate(self):
        artifact, _ = fixture()
        for missing in ('home', 'local', 'same_pool'):
            state = RatingState.from_dict(artifact['state'])
            if missing == 'home':
                del state.homes['Synthetic 1']
            elif missing == 'local':
                del state.local_ratings['R1']['Synthetic 1']
            else:
                state.homes['Synthetic 1'] = 'R0'
                state.local_ratings['R0']['Synthetic 1'] = 1500
            state.apply_batch([dict(team_a='Synthetic 0', team_b='Synthetic 1', result_a=1, use_region_calibration=True)])
            self.assertEqual(sum(state.calibration_games.values()), 0)
            self.assertEqual(sum(abs(x) for x in state.offsets.values()), 0)

    def test_same_timestamp_domestic_does_not_backfill_home(self):
        state = RatingState()
        state.apply_batch([
            dict(team_a='A', team_b='C', result_a=1, use_region_calibration=True),
            dict(team_a='A', team_b='B', result_a=1, rating_pool='R1', updates_home_pool=True, use_local_elo=True),
            dict(team_a='C', team_b='D', result_a=0, rating_pool='R2', updates_home_pool=True, use_local_elo=True),
        ])
        self.assertEqual(state.offsets, {'R1': 0.0, 'R2': 0.0})
        self.assertEqual(state.homes['A'], 'R1')
        self.assertEqual(state.local_ratings['R1']['A'], 1512)
        with self.assertRaises(ValueError):
            state.apply_batch([dict(team_a='A', team_b='B', result_a=1, rating_pool=pool,
                                    updates_home_pool=True, use_local_elo=True) for pool in ('R1','R2')])

    def test_cutoff_and_package_validation(self):
        artifact, _ = fixture()
        for key in ('model_trained_through', 'last_history_timestamp'):
            bad = copy.deepcopy(artifact)
            bad[key] = bad['cutoff_exclusive']
            with self.assertRaises(ValueError):
                validate_artifact(bad)
        bad = copy.deepcopy(artifact)
        bad['state']['global_ratings']['Synthetic 0'] = float('inf')
        with self.assertRaises(ValueError):
            validate_artifact(bad)

    def test_fair_bracket_probabilities_and_reproducibility(self):
        artifact, session = fixture()
        original = json.dumps([artifact, session], sort_keys=True)
        result = simulate(artifact, session, simulations=100_000, seed=42)
        self.assertEqual(result, simulate(artifact, session, simulations=100_000, seed=42))
        self.assertEqual(original, json.dumps([artifact, session], sort_keys=True))
        for row in result['teams']:
            self.assertLess(abs(row['title_probability'] - .125), .006)
            self.assertLessEqual(row['title_probability'], row['final_probability'])
            self.assertLessEqual(row['final_probability'], row['semifinal_probability'])
        self.assertAlmostEqual(sum(r['title_probability'] for r in result['teams']), 1)
        self.assertAlmostEqual(sum(r['final_probability'] for r in result['teams']), 2)
        self.assertAlmostEqual(sum(r['semifinal_probability'] for r in result['teams']), 4)

    def test_partial_score_monte_carlo_matches_exact(self):
        artifact, session = fixture()
        for winner in ['Synthetic 0','Synthetic 1','Synthetic 0']:
            session = record(artifact, session, 'QF1', winner)
        result = simulate(artifact, session, simulations=100_000, seed=9)
        expected = result['series'][0]['series_probability_a']
        actual = next(r['semifinal_probability'] for r in result['teams'] if r['team']=='Synthetic 0')
        self.assertLess(abs(actual-expected), .006)
        self.assertEqual(result['series'][0]['score'], [2,1])

    def test_complete_bracket_is_fixed_and_eliminated_teams_stay_out(self):
        artifact, session = fixture()
        for series in ('QF1','QF2','QF3','QF4','SF1','SF2','F'):
            _, details, _ = replay_session(artifact, session)
            winner = next(d['teams'][0] for d in details if d['series']==series)
            for _ in range(3):
                session = record(artifact, session, series, winner)
        result = simulate(artifact, session, simulations=1000)
        self.assertEqual(result['teams'][0]['team'], 'Synthetic 0')
        self.assertEqual(result['teams'][0]['title_probability'], 1)
        self.assertEqual(sum(row['title_probability'] for row in result['teams'][1:]), 0)
        with self.assertRaises(ValueError):
            record(artifact, session, 'F', 'Synthetic 0')

    def test_bad_results_duplicate_ids_and_unqualified_rounds_rejected(self):
        artifact, session = fixture()
        for series, winner in [('SF1','Synthetic 0'),('QF1','Synthetic 7'),('bad','Synthetic 0')]:
            with self.assertRaises(ValueError):
                record(artifact, session, series, winner)
        session = record(artifact, session, 'QF1', 'Synthetic 0')
        duplicate = copy.deepcopy(session)
        duplicate['events'].append(copy.deepcopy(duplicate['events'][0]))
        with self.assertRaises(ValueError):
            replay_session(artifact, duplicate)
        with self.assertRaises(ValueError):
            add_result(artifact, session, event_id='early', series='QF1', winner='Synthetic 0', timestamp='2019-01-01T00:00:00Z')
        bad = copy.deepcopy(session)
        bad['events'][0]['game_number'] = 2
        with self.assertRaises(ValueError):
            replay_session(artifact,bad)

    def test_replay_resume_undo_and_artifact_mismatch(self):
        artifact, initial = fixture()
        session = record(artifact, initial, 'QF1', 'Synthetic 0')
        resumed = json.loads(json.dumps(session))
        self.assertEqual(replay_session(artifact,session)[0].as_dict(), replay_session(artifact,resumed)[0].as_dict())
        resumed['events'].pop()
        self.assertEqual(replay_session(artifact,resumed)[0].as_dict(), artifact['state'])
        altered = copy.deepcopy(artifact)
        altered['predictor']['regional_weight'] = .1
        with self.assertRaises(ValueError):
            replay_session(altered,session)

    def test_invalid_brackets_and_run_settings(self):
        artifact, session = fixture()
        bad = copy.deepcopy(session['bracket'])
        bad['quarterfinals'][0][0] = bad['quarterfinals'][0][1]
        with self.assertRaises(ValueError):
            new_session(artifact,bad)
        for kwargs in ({'simulations':0},{'simulations':100.5},{'seed':-1}):
            with self.assertRaises(ValueError):
                simulate(artifact,session,**kwargs)


if __name__ == '__main__':
    unittest.main()
