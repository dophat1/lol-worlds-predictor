"""Score aligned prediction ledgers without fitting or selecting a model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    version: str
    features: tuple[str, ...] = ()
    settings: dict = field(default_factory=dict)
    fallback_model_id: str | None = None
    selection_period: str | None = None

    def as_dict(self):
        return asdict(self)


MATCH_COLUMNS = ["match_id", "period", "date", "league", "team_a", "team_b", "result_a", "international_scope"]
PREDICTION_COLUMNS = [
    "match_id", "period", "model_id", "probability_a", "prediction_timestamp", "information_cutoff_exclusive",
    "model_trained_through", "configuration_selected_through", "native_available", "used_fallback",
    "parameter_selection_reused", "evaluation_mode",
]


def utc_column(series, *, nullable=False):
    """Normalize explicit offset-aware times; do not silently localize naive times."""
    if not nullable and series.isna().any():
        raise ValueError("Required timestamps are missing.")
    for value in series.dropna().unique():
        if pd.Timestamp(value).tzinfo is None:
            raise ValueError("Timestamps must explicitly include a timezone.")
    return pd.to_datetime(series, utc=True, errors="raise", format="mixed")


def _booleans(frame, columns):
    for name in columns:
        if not pd.api.types.is_bool_dtype(frame[name]) or frame[name].isna().any():
            raise ValueError(f"{name} must contain actual booleans.")


def validate_predictions(matches, predictions, models, evaluation_mode="online_elo"):
    if not set(MATCH_COLUMNS) <= set(matches) or not set(PREDICTION_COLUMNS) <= set(predictions):
        raise ValueError("Incomplete match or prediction schema.")
    if evaluation_mode not in {"online_elo", "frozen_pre_tournament"}:
        raise ValueError("Unknown evaluation mode.")
    matches, predictions = matches.copy(), predictions.copy()
    if matches.empty or predictions.empty:
        raise ValueError("Cannot evaluate empty inputs.")
    if not models or len({m.model_id for m in models}) != len(models):
        raise ValueError("Model IDs must be nonempty and unique.")
    if any(not m.model_id.strip() or not m.version.strip() for m in models):
        raise ValueError("Every model requires an ID and version.")
    ids = {m.model_id for m in models}
    if any(m.fallback_model_id and m.fallback_model_id not in ids for m in models):
        raise ValueError("A fallback references an unregistered model.")
    for name in ["match_id", "period", "league", "team_a", "team_b"]:
        if matches[name].isna().any() or matches[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing or blank match field: {name}.")
    if matches["match_id"].duplicated().any() or matches["team_a"].eq(matches["team_b"]).any():
        raise ValueError("Match IDs must be unique and opponents distinct.")
    if not matches["result_a"].isin([0, 1]).all():
        raise ValueError("Outcomes must be binary.")
    matches["result_a"] = matches["result_a"].astype(int)
    matches["date"] = utc_column(matches["date"])
    ranges = matches.groupby("period")["date"].agg(["min", "max"]).sort_values("min")
    if (ranges["max"].shift() >= ranges["min"]).any():
        raise ValueError("Evaluation periods overlap or split an equal timestamp.")
    _booleans(matches, ["international_scope"])
    _booleans(predictions, ["native_available", "used_fallback", "parameter_selection_reused"])
    if predictions["evaluation_mode"].isna().any() or not predictions["evaluation_mode"].eq(evaluation_mode).all():
        raise ValueError("Do not mix evaluation modes in one run.")
    if set(predictions["model_id"]) != ids:
        raise ValueError("Prediction and registry model IDs differ.")
    if predictions.duplicated(["model_id", "match_id"]).any():
        raise ValueError("Duplicate model/match predictions.")
    expected = set(zip(matches["period"], matches["match_id"]))
    for model_id, rows in predictions.groupby("model_id"):
        if set(zip(rows["period"], rows["match_id"])) != expected:
            raise ValueError(f"{model_id} does not predict exactly the common match population.")
    probabilities = pd.to_numeric(predictions["probability_a"], errors="raise").to_numpy(float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Every match needs a finite probability in [0,1], with explicit fallback if necessary.")
    predictions["probability_a"] = probabilities
    for name in ["prediction_timestamp", "information_cutoff_exclusive", "model_trained_through", "configuration_selected_through"]:
        predictions[name] = utc_column(predictions[name], nullable=name in {"model_trained_through", "configuration_selected_through"})
    if set(predictions) & {"date", "league", "team_a", "team_b", "result_a", "international_scope"}:
        raise ValueError("Keep outcomes and match identities in the separate canonical match table.")
    timing = predictions.groupby("match_id")[["prediction_timestamp", "information_cutoff_exclusive"]].nunique()
    if timing.gt(1).any().any():
        raise ValueError("Models must use identical forecast timing on each common match.")
    merged = predictions.merge(matches[MATCH_COLUMNS], on=["match_id", "period"], validate="many_to_one")
    if (merged["prediction_timestamp"] > merged["date"]).any():
        raise ValueError("Prediction information time is after the match.")
    if (merged["information_cutoff_exclusive"] > merged["prediction_timestamp"]).any():
        raise ValueError("Information cutoff is after prediction time.")
    fitted = merged["model_trained_through"].notna()
    if (merged.loc[fitted, "model_trained_through"] >= merged.loc[fitted, "information_cutoff_exclusive"]).any():
        raise ValueError("Fitted coefficients use information at/after prediction cutoff.")
    if evaluation_mode == "online_elo" and not merged["information_cutoff_exclusive"].eq(merged["date"]).all():
        raise ValueError("Online Elo records require the logical pre-match cutoff.")
    selection = {m.model_id: m.selection_period for m in models}
    valid_reuse = merged["period"].eq(merged["model_id"].map(selection)) & merged["period"].eq("validation")
    if (merged["parameter_selection_reused"] & ~valid_reuse).any():
        raise ValueError("Selection reuse is permitted only as an explicitly labeled validation diagnostic.")
    selected = merged["configuration_selected_through"].notna()
    future_selection = selected & (merged["configuration_selected_through"] >= merged["information_cutoff_exclusive"])
    if (future_selection & ~merged["parameter_selection_reused"]).any():
        raise ValueError("Configuration selection uses future outcomes without a validation-reuse label.")
    if (merged["native_available"] & merged["used_fallback"]).any() or (~merged["native_available"] & ~merged["used_fallback"]).any():
        raise ValueError("Declare native prediction availability and fallback consistently.")
    for model in models:
        rows = predictions.loc[predictions["model_id"].eq(model.model_id)]
        if model.fallback_model_id:
            reference = predictions.loc[predictions["model_id"].eq(model.fallback_model_id)].set_index("match_id")["probability_a"]
            fallback = rows.loc[rows["used_fallback"]]
            if not np.allclose(fallback["probability_a"], fallback["match_id"].map(reference), atol=1e-12, rtol=1e-12):
                raise ValueError("Declared fallback probabilities do not match the fallback model.")
        elif rows["used_fallback"].any():
            raise ValueError("Fallback use requires a registered fallback model.")
    return matches.sort_values(["date", "match_id"]).reset_index(drop=True), predictions.sort_values(["period", "model_id", "match_id"]).reset_index(drop=True)


def per_game_scores(outcomes, probabilities):
    raw_y, p = np.asarray(outcomes), np.asarray(probabilities, dtype=float)
    if raw_y.ndim != 1 or p.ndim != 1 or len(raw_y) != len(p):
        raise ValueError("Outcomes and probabilities must be aligned one-dimensional arrays.")
    if not np.isin(raw_y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary outcomes and finite probabilities in [0,1].")
    y = raw_y.astype(int)
    clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {
        "accuracy": ((p >= .5) == y).astype(float),
        "log_loss": -(y * np.log(clipped) + (1-y) * np.log1p(-clipped)),
        "brier": (p-y)**2,
    }


def calibration_bins(outcomes, probabilities, bins=10):
    y, p = np.asarray(outcomes, dtype=int), np.asarray(probabilities, dtype=float)
    # Right-closed bins match the existing notebook's pd.cut convention.
    indices = np.clip(np.searchsorted(np.linspace(0, 1, bins + 1), p, side="left") - 1, 0, bins - 1)
    result = []
    for index in range(bins):
        mask = indices == index
        result.append({"bin": index, "lower": index / bins, "upper": (index+1) / bins,
                       "matches": int(mask.sum()), "predicted_probability": float(p[mask].mean()) if mask.any() else np.nan,
                       "observed_win_rate": float(y[mask].mean()) if mask.any() else np.nan})
    return pd.DataFrame(result)


def probability_metrics(outcomes, probabilities, bins=10):
    per_game = per_game_scores(outcomes, probabilities)
    y, p = np.asarray(outcomes, dtype=int), np.asarray(probabilities, dtype=float)
    if len(y) == 0:
        return {"matches": 0, **dict.fromkeys(["accuracy", "log_loss", "brier", "roc_auc", "ece"], np.nan)}
    scores = {name: float(values.mean()) for name, values in per_game.items()}
    scores["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    table = calibration_bins(y, p, bins)
    nonempty = table.loc[table["matches"].gt(0)]
    scores["ece"] = float((nonempty["matches"] * (nonempty["predicted_probability"] - nonempty["observed_win_rate"]).abs()).sum() / len(y))
    return {"matches": len(y), **scores}


def week_blocks(dates):
    normalized = dates.dt.normalize()
    return normalized - pd.to_timedelta(normalized.dt.day_of_week, unit="D")


def paired_differences(outcomes, candidate, baseline, blocks, *, resamples=1000, seed=2026, confidence=.95):
    if type(resamples) is not int or not 100 <= resamples <= 10_000:
        raise ValueError("Bootstrap repetitions must be an integer between 100 and 10,000.")
    if type(seed) is not int or not 0 <= seed <= 2**32-1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap seed or confidence level.")
    if not (len(outcomes) == len(candidate) == len(baseline) == len(blocks)):
        raise ValueError("Paired bootstrap inputs do not align.")
    codes, unique = pd.factorize(pd.Series(blocks), sort=True)
    if (codes < 0).any():
        raise ValueError("Bootstrap blocks cannot be missing.")
    count = len(unique)
    weights = np.bincount(codes, minlength=count)
    indices = np.random.default_rng(seed).integers(0, count, size=(resamples, count)) if count >= 2 else None
    candidate_scores, baseline_scores = per_game_scores(outcomes, candidate), per_game_scores(outcomes, baseline)
    records = []
    for name in ("accuracy", "log_loss", "brier"):
        differences = candidate_scores[name] - baseline_scores[name]
        delta = float(differences.mean()) if len(differences) else np.nan
        low = high = np.nan
        if indices is not None:
            totals = np.bincount(codes, weights=differences, minlength=count)
            samples = totals[indices].sum(axis=1) / weights[indices].sum(axis=1)
            low, high = np.quantile(samples, [(1-confidence)/2, 1-(1-confidence)/2])
        records.append({"metric": name, "candidate_minus_baseline": delta, "ci_lower": low, "ci_upper": high,
                        "matches": len(differences), "blocks": count, "resamples": resamples, "confidence": confidence,
                        "interval_status": "insufficient_blocks" if count < 2 else "few_blocks" if count < 10 else "diagnostic",
                        "better_direction": "positive" if name == "accuracy" else "negative"})
    return pd.DataFrame(records)


def evaluate_predictions(matches, predictions, models, *, baseline_model_id="global_logistic_v1",
                         resamples=1000, seed=2026, confidence=.95, bins=10, evaluation_mode="online_elo"):
    if type(bins) is not int or not 2 <= bins <= 50:
        raise ValueError("Use between 2 and 50 probability bins.")
    # Validate bootstrap configuration even for an empty comparison scope.
    paired_differences([], [], [], [], resamples=resamples, seed=seed, confidence=confidence)
    matches, predictions = validate_predictions(matches, predictions, models, evaluation_mode)
    if baseline_model_id not in {m.model_id for m in models}:
        raise ValueError("The reference baseline is not registered.")
    probabilities = predictions.pivot(index="match_id", columns="model_id", values="probability_a")
    native = predictions.pivot(index="match_id", columns="model_id", values="native_available")
    fallback = predictions.pivot(index="match_id", columns="model_id", values="used_fallback")
    reused = predictions.pivot(index="match_id", columns="model_id", values="parameter_selection_reused")
    matches = matches.set_index("match_id", drop=False)
    scopes = {"all": pd.Series(True, index=matches.index), "international": matches["international_scope"],
              "msi_and_worlds": matches["league"].isin(["MSI", "WLDs"]), "worlds": matches["league"].eq("WLDs")}
    common = native.reindex(matches.index).all(axis=1)
    metric_rows, coverage_rows, calibration_tables, difference_tables, temporal_rows = [], [], [], [], []
    for period in matches["period"].drop_duplicates():
        for scope, scope_mask in scopes.items():
            scoped = matches.loc[matches["period"].eq(period) & scope_mask]
            for population, group in [("full_scope", scoped), ("common_native", scoped.loc[common.reindex(scoped.index)])]:
                keys = {"period": period, "scope": scope, "population": population, "evaluation_mode": evaluation_mode}
                ids = group.index
                y = group["result_a"].to_numpy(int)
                for model in models:
                    p = probabilities.loc[ids, model.model_id].to_numpy(float)
                    model_keys = {**keys, "model_id": model.model_id}
                    metric_rows.append({**model_keys, **probability_metrics(y, p, bins),
                                        "selection_reused": bool(reused.loc[ids, model.model_id].any())})
                    available = int(native.loc[ids, model.model_id].sum())
                    coverage_rows.append({**model_keys, "matches": len(ids), "scope_matches_before_common_filter": len(scoped),
                                          "native_predictions": available, "fallback_predictions": int(fallback.loc[ids, model.model_id].sum()),
                                          "native_share": available/len(ids) if len(ids) else np.nan})
                    calibration_tables.append(calibration_bins(y, p, bins).assign(**model_keys))
                    if model.model_id != baseline_model_id:
                        difference_tables.append(paired_differences(
                            y, p, probabilities.loc[ids, baseline_model_id].to_numpy(float), week_blocks(group["date"]),
                            resamples=resamples, seed=seed, confidence=confidence,
                        ).assign(**model_keys, baseline_model_id=baseline_model_id, block_definition="UTC_calendar_week_Monday"))
                if population == "full_scope":
                    for year, yearly in group.groupby(group["date"].dt.year):
                        for model in models:
                            temporal_rows.append({**keys, "calendar_year": int(year), "model_id": model.model_id,
                                                  **probability_metrics(yearly["result_a"], probabilities.loc[yearly.index, model.model_id], bins),
                                                  "native_predictions": int(native.loc[yearly.index, model.model_id].sum()),
                                                  "fallback_predictions": int(fallback.loc[yearly.index, model.model_id].sum())})
    return {"metrics": pd.DataFrame(metric_rows), "coverage": pd.DataFrame(coverage_rows),
            "calibration": pd.concat(calibration_tables, ignore_index=True),
            "paired_differences": pd.concat(difference_tables, ignore_index=True) if difference_tables else pd.DataFrame(),
            "temporal_metrics": pd.DataFrame(temporal_rows)}
