"""Standalone prediction contracts, using explicitly synthetic fixtures."""

import copy
import json
import math
import unittest

from src.prediction import Predictor, RatingState
from src.predict_match import forecast_match


def synthetic_artifact():
    state = RatingState(
        global_ratings={"Synthetic A": 1600.0, "Synthetic B": 1500.0},
        local_ratings={"R1": {"Synthetic A": 1550.0}, "R2": {"Synthetic B": 1525.0}},
        homes={"Synthetic A": "R1", "Synthetic B": "R2"}, offsets={"R1": 30.0, "R2": -30.0},
        local_games={"R1": {"Synthetic A": 10}, "R2": {"Synthetic B": 12}},
        calibration_games={"R1": 6, "R2": 6},
    )
    return {"schema_version": 1, "kind": "trained_model", "cutoff_exclusive": "2020-01-02T00:00:00Z",
            "model_trained_through": "2019-12-31T00:00:00Z", "last_history_timestamp": "2020-01-01T00:00:00Z",
            "predictor": Predictor(.005, (.003, .007), .8).as_dict(), "state": state.as_dict(),
            "metadata": {"description": "Synthetic mathematical test fixture; not actual match data."}}


class PredictorTests(unittest.TestCase):
    def test_game_and_series_symmetry(self):
        artifact = synthetic_artifact()
        forward = forecast_match(artifact, "Synthetic A", "Synthetic B", 2, 1)
        reverse = forecast_match(artifact, "Synthetic B", "Synthetic A", 1, 2)
        for column in ("probability_a", "global_probability_a", "regional_probability_a"):
            self.assertAlmostEqual(forward["prediction"][column] + reverse["prediction"][column], 1)
        self.assertAlmostEqual(forward["best_of_five"]["probability_a"] + reverse["best_of_five"]["probability_a"], 1)
        self.assertEqual(forward["teams"]["a"]["adjusted_strength"], 1580)
        self.assertEqual(forward["prediction"]["effective_regional_weight"], .8)

    def test_global_scope_never_applies_regional_model(self):
        artifact = synthetic_artifact()
        report = forecast_match(artifact, "Synthetic A", "Synthetic B", use_international_model=False)
        self.assertEqual(report["prediction"]["probability_a"], report["prediction"]["global_probability_a"])
        self.assertIsNone(report["prediction"]["regional_probability_a"])
        self.assertEqual(report["prediction"]["fallback_reason"], "outside_international_model_scope")
        self.assertEqual(report["prediction"]["effective_regional_weight"], 0)
        with self.assertRaises(ValueError):
            forecast_match(artifact, "Synthetic A", "Synthetic B", use_international_model="false")

    def test_unknown_home_and_missing_local_ratings_are_not_zero(self):
        for missing in ("home", "rating"):
            artifact = synthetic_artifact()
            if missing == "home":
                del artifact["state"]["homes"]["Synthetic A"]
            else:
                del artifact["state"]["local_ratings"]["R1"]["Synthetic A"]
            report = forecast_match(artifact, "Synthetic A", "Synthetic B")
            self.assertTrue(report["prediction"]["fallback"])
            self.assertEqual(report["prediction"]["fallback_reason"], "missing_home_or_local_rating")
            self.assertIsNone(report["teams"]["a"]["local_elo"])
            self.assertIsNone(report["teams"]["a"]["adjusted_strength"])

    def test_query_does_not_apply_scores_as_rating_updates(self):
        artifact = synthetic_artifact()
        before = copy.deepcopy(artifact)
        first = forecast_match(artifact, "Synthetic A", "Synthetic B")
        partial = forecast_match(artifact, "Synthetic A", "Synthetic B", 2, 0)
        self.assertEqual(first["prediction"], partial["prediction"])
        self.assertGreater(partial["best_of_five"]["probability_a"], first["best_of_five"]["probability_a"])
        self.assertEqual(artifact, before)

    def test_completed_series_and_invalid_scores(self):
        artifact = synthetic_artifact()
        report = forecast_match(artifact, "Synthetic A", "Synthetic B", 3, 2)
        self.assertEqual(report["best_of_five"]["probability_a"], 1)
        self.assertTrue(report["best_of_five"]["completed"])
        for score in ((3, 3), (-1, 0), (4, 0)):
            with self.assertRaises(ValueError):
                forecast_match(artifact, "Synthetic A", "Synthetic B", *score)

    def test_json_roundtrip_and_invalid_inputs(self):
        artifact = synthetic_artifact()
        report = forecast_match(artifact, "Synthetic A", "Synthetic B")
        self.assertEqual(report, json.loads(json.dumps(report, allow_nan=False)))
        for teams in (("Synthetic A", "Synthetic A"), ("Unknown", "Synthetic B")):
            with self.assertRaises(ValueError):
                forecast_match(artifact, *teams)
        bad = copy.deepcopy(artifact)
        bad["model_trained_through"] = bad["cutoff_exclusive"]
        with self.assertRaises(ValueError):
            forecast_match(bad, "Synthetic A", "Synthetic B")

    def test_blend_matches_direct_logistic_formula(self):
        artifact = synthetic_artifact()
        report = forecast_match(artifact, "Synthetic A", "Synthetic B")
        global_p = 1 / (1 + math.exp(-.005 * 100))
        regional_p = 1 / (1 + math.exp(-(.003 * 25 + .007 * 60)))
        self.assertAlmostEqual(report["prediction"]["probability_a"], .2 * global_p + .8 * regional_p)


if __name__ == "__main__":
    unittest.main()
