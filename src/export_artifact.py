"""Export a trusted local notebook model plus strictly pre-cutoff rating state.

Usage: python -m src.export_artifact --cutoff 2026-10-27T00:00:00Z
This example is a source-data snapshot, not a claim about official event dates.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .prediction import Predictor, RatingState, utc, validate_artifact


def _coefficients(pipeline, size):
    scaler = pipeline.named_steps["standardscaler"]
    model = pipeline.named_steps["logisticregression"]
    if scaler.with_mean or model.fit_intercept or list(model.classes_) != [0, 1]:
        raise ValueError("Expected the notebook's no-intercept symmetric model.")
    coefficient = model.coef_[0] / scaler.scale_
    if len(coefficient) != size or not np.isfinite(coefficient).all():
        raise ValueError("Unexpected model features.")
    return coefficient.tolist()


def _same(actual, expected, field, match_id):
    if actual is None:
        equal = pd.isna(expected)
    else:
        equal = not pd.isna(expected) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7)
    if not equal:
        raise ValueError(f"Historical replay differs for {match_id}: {field}.")


def export_artifact(root, cutoff, output):
    root, output = Path(root), Path(output)
    cutoff_time = utc(cutoff)
    # Load only our trusted local model file, never a browser-uploaded pickle.
    bundle = joblib.load(root / "models" / "international_logistic_blend.joblib")
    metadata = bundle["metadata"]
    if utc(metadata["trained_through"]) >= cutoff_time:
        raise ValueError("The fitted model contains outcomes at/after this cutoff. Train an earlier model first.")
    if metadata["global_features"] != ["elo_diff"] or metadata["regional_features"] != ["local_elo_diff", "pool_offset_diff"]:
        raise ValueError("Unexpected feature definitions.")
    paths = {
        "calibrated_elo_features_all_years.csv": root / "data/processed/calibrated_elo_features_all_years.csv",
        "elo_features_all_years.csv": root / "data/processed/elo_features_all_years.csv",
        "chronological_splits.csv": root / "results/baseline/chronological_splits.csv",
    }
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["input_sha256"].get(name):
            raise ValueError(f"{name} no longer matches the model's recorded input. Regenerate a consistent model/export.")
    history = pd.read_csv(paths["calibrated_elo_features_all_years.csv"])
    history["date"] = pd.to_datetime(history["date"], utc=True)
    if not history["match_id"].is_unique:
        raise ValueError("Duplicate match IDs in the feature history.")
    for flag in ("updates_home_pool", "use_local_elo", "use_region_calibration"):
        if not pd.api.types.is_bool_dtype(history[flag]):
            raise ValueError(f"Expected a boolean {flag} column.")
    global_history = pd.read_csv(paths["elo_features_all_years.csv"])
    keys = ["match_id", "team_a", "team_b", "result_a", "league"]
    if not global_history["match_id"].is_unique:
        raise ValueError("Duplicate global match IDs.")
    history = history.merge(global_history[keys + ["pre_elo_a", "pre_elo_b"]],
                            on=keys, validate="one_to_one", how="left")
    if history[["pre_elo_a", "pre_elo_b"]].isna().any().any():
        raise ValueError("Global and regional history do not align.")
    prefix = history.loc[history["date"].lt(cutoff_time)].sort_values(["date", "match_id"])
    if prefix.empty:
        raise ValueError("No history strictly before this cutoff.")
    state = RatingState()
    for _, group in itertools.groupby(prefix.itertuples(index=False), key=lambda row: row.date):
        rows = [row._asdict() for row in group]
        for row in rows:
            for side in ("a", "b"):
                team = row[f"team_{side}"]
                pool = row["rating_pool"] if row["updates_home_pool"] else state.homes.get(team)
                pool = None if pd.isna(pool) else pool
                expected_pool = row[f"home_pool_{side}"]
                if (pool or None) != (None if pd.isna(expected_pool) else expected_pool):
                    raise ValueError("Home history differs from the notebook features.")
                values = {
                    f"pre_elo_{side}": state.global_ratings.get(team, 1500.0),
                    f"pre_local_elo_{side}": state.local_ratings.get(pool, {}).get(team, 1500.0) if pool else None,
                    f"pre_local_games_{side}": state.local_games.get(pool, {}).get(team, 0) if pool else 0,
                    f"pre_pool_offset_{side}": state.offsets.get(pool, 0.0) if pool else None,
                    f"pre_pool_calibration_games_{side}": state.calibration_games.get(pool, 0) if pool else None,
                }
                for name, value in values.items():
                    _same(value, row[name], name, row["match_id"])
        state.apply_batch(rows)
    predictor = Predictor(_coefficients(bundle["global_model"], 1)[0],
                          tuple(_coefficients(bundle["regional_model"], 2)), metadata["selected_regional_weight"])
    # Check portable inference against the original pipelines, including symmetry.
    differences = np.array([[-400., -150.], [0., 0.], [400., 150.]])
    np.testing.assert_allclose(
        [1 / (1 + math.exp(-sum(c * x for c, x in zip(predictor.regional_coefficients, row)))) for row in differences],
        bundle["regional_model"].predict_proba(differences)[:, 1],
    )
    artifact = {
        "schema_version": 1, "kind": "trained_model", "cutoff_exclusive": cutoff_time.isoformat(),
        "model_trained_through": metadata["trained_through"], "last_history_timestamp": prefix["date"].max().isoformat(),
        "predictor": predictor.as_dict(), "state": state.as_dict(),
        "metadata": {"history_games": len(prefix), "historical_replay_verified": True,
                     "input_sha256": metadata["input_sha256"], "model_metadata": metadata,
                     "model_bundle_sha256": hashlib.sha256((root / "models/international_logistic_blend.joblib").read_bytes()).hexdigest(),
                     "assumptions": "Source league labels; exact team names; latest earlier domestic homes; fixed logistic coefficients."},
    }
    validate_artifact(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, help="Exclusive UTC/offset-aware history cutoff.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="models/simulator_artifact.json")
    args = parser.parse_args()
    try:
        artifact = export_artifact(args.root, args.cutoff, args.output)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    print(f"Exported {args.output}: {artifact['metadata']['history_games']:,} earlier games; "
          f"{len(artifact['state']['global_ratings']):,} teams. Last source game: {artifact['last_history_timestamp']}")


if __name__ == "__main__":
    main()
