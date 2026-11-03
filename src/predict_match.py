"""Explain a frozen-snapshot game and best-of-five forecast without the app.

Example matchup query (not an actual match record):
python -m src.predict_match --artifact models/simulator_artifact.json --team-a "Gen.G" --team-b "T1"
"""

from __future__ import annotations

import argparse
import json

from .prediction import Predictor, RatingState, load_artifact, validate_artifact
from .tournament import exact_series_probability


def forecast_match(artifact, team_a, team_b, wins_a=0, wins_b=0, *, use_international_model=True):
    """Return auditable probabilities; never update snapshot state or model."""
    validate_artifact(artifact)
    if artifact["kind"] != "trained_model":
        raise ValueError("This entry point requires a trained-model artifact, not the hypothetical demo.")
    state = RatingState.from_dict(artifact["state"])
    predictor = Predictor(**artifact["predictor"])
    prediction = predictor.predict(state, team_a, team_b, use_international_model=use_international_model)
    probability = prediction["probability_a"]
    series_probability = exact_series_probability(probability, wins_a, wins_b)
    teams = {}
    for side, team in [("a", team_a), ("b", team_b)]:
        pool = state.homes.get(team)
        local = state.local(team)
        offset = state.offsets.get(pool, 0.0) if pool else None
        teams[side] = {
            "team": team, "global_elo": state.global_ratings[team], "home_pool": pool,
            "local_elo": local, "pool_offset": offset,
            "adjusted_strength": local + offset if local is not None and offset is not None else None,
            "prior_domestic_games": state.local_games.get(pool, {}).get(team) if local is not None else None,
            "prior_pool_calibration_games": state.calibration_games.get(pool, 0) if pool else None,
        }
    reason = None
    if not use_international_model:
        reason = "outside_international_model_scope"
    elif prediction["fallback"]:
        reason = "missing_home_or_local_rating"
    return {
        "schema_version": 1, "forecast_mode": "frozen_snapshot_matchup_query",
        "cutoff_exclusive": artifact["cutoff_exclusive"],
        "model_trained_through": artifact["model_trained_through"],
        "last_history_timestamp": artifact["last_history_timestamp"],
        "teams": teams, "use_international_model": use_international_model,
        "prediction": {**prediction, "probability_b": 1 - probability,
                       "fallback_reason": reason,
                       "effective_regional_weight": 0.0 if prediction["fallback"] else predictor.regional_weight},
        "best_of_five": {"wins_a": wins_a, "wins_b": wins_b,
                         "probability_a": series_probability, "probability_b": 1 - series_probability,
                         "completed": wins_a == 3 or wins_b == 3},
        "assumptions": "Supplied matchup and score, not an actual match record. Independent remaining games with constant probability; no simulated rating updates.",
        "evaluation_status": artifact["metadata"].get("model_metadata", {}).get(
            "test_status", "Development model; no untouched-final-performance claim."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--wins-a", type=int, default=0, help="Assumed/observed score; this command does not record results.")
    parser.add_argument("--wins-b", type=int, default=0)
    parser.add_argument("--scope", choices=("international", "global"), default="international",
                        help="Use international blend, or global logistic only outside that scope.")
    args = parser.parse_args()
    try:
        report = forecast_match(load_artifact(args.artifact), args.team_a, args.team_b, args.wins_a, args.wins_b,
                                use_international_model=args.scope == "international")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(1, f"Prediction failed: {exc}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
