"""Versioned, score-only evaluation of the predictions saved by notebooks 03/05.

Run: python -m src.evaluate --root .
No model is trained, selected, or modified by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn

from .evaluation import ModelSpec, evaluate_predictions, utc_column


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def report_staging(parent):
    """Use normal parent permissions, not TemporaryDirectory's private ACL.

    Windows retains directory ACLs when an artifact folder is renamed. A
    private tempfile ancestor would make published reports unreadable by the
    workspace's normal user when generated under the sandbox account.
    """
    parent = Path(parent).resolve()
    folder = parent / (".evaluation-build-" + uuid4().hex)
    folder.mkdir()
    try:
        yield folder
    finally:
        if folder.exists():
            if folder.is_symlink() or folder.resolve().parent != parent:
                raise ValueError("Refusing to clean a staging directory outside its output parent.")
            shutil.rmtree(folder)


def _close(actual, expected, description):
    if not np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), atol=1e-12, rtol=1e-12, equal_nan=True):
        raise ValueError(description)


def load_saved_suite(root):
    root = Path(root).resolve()
    files = {
        "truth": root / "data/processed/matches_clean.csv",
        "global_features": root / "data/processed/elo_features_all_years.csv",
        "regional_features": root / "data/processed/calibrated_elo_features_all_years.csv",
        "splits": root / "results/baseline/chronological_splits.csv",
        "metadata": root / "results/international_model/model_metadata.json",
        "validation": root / "results/international_model/validation_predictions.csv",
        "development_test": root / "results/international_model/test_predictions.csv",
        "baseline_test": root / "results/baseline/test_predictions.csv",
        "validation_metrics": root / "results/international_model/validation_metrics.csv",
        "test_metrics": root / "results/international_model/test_metrics.csv",
    }
    for path in files.values():
        if not path.is_file():
            raise ValueError(f"Missing {path.relative_to(root)}. Run the analysis notebooks first.")
    hashes = {path.relative_to(root).as_posix(): file_hash(path) for path in files.values()}
    metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
    for key in ("global_features", "regional_features", "splits"):
        if metadata["input_sha256"].get(files[key].name) != hashes[files[key].relative_to(root).as_posix()]:
            raise ValueError("Model metadata and feature/split files no longer describe the same snapshot.")
    if metadata["global_features"] != ["elo_diff"] or metadata["regional_features"] != ["local_elo_diff", "pool_offset_diff"]:
        raise ValueError("This saved-model adapter expects the current rating-based feature definitions.")
    weight = float(metadata["selected_regional_weight"])
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Invalid saved blend weight.")
    splits = pd.read_csv(files["splits"])
    if splits["split"].duplicated().any() or set(splits["split"]) != {"training", "validation", "testing"}:
        raise ValueError("Expected unique training, validation, and testing boundaries.")
    for column in ("start", "end"):
        splits[column] = utc_column(splits[column])
    splits = splits.set_index("split")
    if not (splits.loc["training", "end"] < splits.loc["validation", "start"] <= splits.loc["validation", "end"] < splits.loc["testing", "start"]):
        raise ValueError("Split periods overlap or are out of order.")
    if pd.Timestamp(metadata["trained_through"]) != splits.loc["validation", "end"]:
        raise ValueError("Final coefficient training endpoint differs from the saved development split.")
    identity = ["match_id", "date", "league", "team_a", "team_b", "result_a"]
    truth = pd.read_csv(files["truth"])
    truth["date"] = utc_column(truth["date"])
    if truth["match_id"].duplicated().any():
        raise ValueError("Canonical match history contains duplicate IDs.")
    history = pd.read_csv(files["regional_features"])
    history["date"] = utc_column(history["date"])
    try:
        pd.testing.assert_frame_equal(history[identity].sort_values("match_id").reset_index(drop=True),
                                      truth[identity].sort_values("match_id").reset_index(drop=True), check_dtype=False)
    except AssertionError as exc:
        raise ValueError("Regional features and canonical match history differ.") from exc
    if len(truth.loc[truth["date"].lt(splits.loc["testing", "start"])]) != metadata["final_global_training_games"]:
        raise ValueError("Global training count differs from earlier canonical history.")
    for flag in ("use_international_model", "home_known_a", "home_known_b"):
        if not pd.api.types.is_bool_dtype(history[flag]):
            raise ValueError("Regional history flags must be boolean.")
    region_available = (history["use_international_model"] & history["home_known_a"] & history["home_known_b"]
                        & np.isfinite(history["local_elo_diff"])
                        & np.isfinite(history["pre_pool_offset_a"] - history["pre_pool_offset_b"]))
    regional_training_ends = {}
    for period, start, expected_count in [
        ("validation", splits.loc["validation", "start"], metadata["validation_regional_training_games"]),
        ("development_test", splits.loc["testing", "start"], metadata["final_regional_training_games"]),
    ]:
        training = history.loc[region_available & history["date"].lt(start)]
        if len(training) != expected_count or training.empty:
            raise ValueError("Saved regional training count differs from earlier eligible history.")
        regional_training_ends[period] = training["date"].max()

    settings = metadata["fixed_settings"]
    models = [
        ModelSpec("constant_50_v1", "Constant 50%", "1", settings={"probability": .5, "accuracy_tie_rule": "predict_A"}),
        ModelSpec("global_elo_v1", "Global Elo", "1", ("elo_diff",), {"initial_elo": 1500, "K": 24, "scale": 400}),
        ModelSpec("global_logistic_v1", "Global logistic", "1", ("elo_diff",), settings),
        ModelSpec("regional_logistic_v1", "Regional logistic", "1", ("local_elo_diff", "pool_offset_diff"), settings,
                  fallback_model_id="global_logistic_v1"),
        ModelSpec("blend_v1", "Validation-selected blend", "1", ("elo_diff", "local_elo_diff", "pool_offset_diff"),
                  {"regional_weight": weight, "candidate_weights": metadata["blend_weight_candidates"]},
                  fallback_model_id="global_logistic_v1", selection_period="validation"),
    ]
    matches, ledgers, period_metadata = [], [], {}
    saved_frames = {}
    for period, split_name in [("validation", "validation"), ("development_test", "testing")]:
        frame = pd.read_csv(files[period])
        frame["date"] = utc_column(frame["date"])
        start, end = splits.loc[split_name, ["start", "end"]]
        expected = truth.loc[truth["date"].between(start, end), identity].sort_values("match_id").reset_index(drop=True)
        if len(expected) != int(splits.loc[split_name, "matches"]) or frame["match_id"].duplicated().any():
            raise ValueError("Period match counts or uniqueness disagree with the split.")
        try:
            pd.testing.assert_frame_equal(expected, frame[identity].sort_values("match_id").reset_index(drop=True), check_dtype=False)
        except AssertionError as exc:
            raise ValueError("Saved predictions do not match canonical identities, dates, or outcomes.") from exc
        if frame["date"].min() != start or frame["date"].max() != end:
            raise ValueError("Prediction date coverage differs from saved boundaries.")
        for flag in ("international_scope", "regional_available", "home_known_a", "home_known_b"):
            if not pd.api.types.is_bool_dtype(frame[flag]):
                raise ValueError("Prediction availability flags must be boolean.")
        if not frame["international_scope"].eq(frame["league"].isin(metadata["scope"])).all():
            raise ValueError("International scope disagrees with the declared source labels.")
        available = (frame["international_scope"] & frame["home_known_a"] & frame["home_known_b"]
                     & np.isfinite(frame[["local_elo_diff", "pool_offset_diff"]]).all(axis=1))
        if not frame["regional_available"].eq(available).all():
            raise ValueError("Native regional availability differs from finite known-home eligibility.")
        if frame.loc[available, "regional_logistic_probability_a"].isna().any() or frame.loc[~available, "regional_logistic_probability_a"].notna().any():
            raise ValueError("Native regional probabilities do not respect their eligibility.")
        fallback = frame["regional_logistic_probability_a"].where(available, frame["global_logistic_probability_a"])
        _close(frame["regional_with_global_fallback"], fallback, "Saved regional fallback is inconsistent.")
        _close(frame["blend_probability_a"], (1-weight)*frame["global_logistic_probability_a"] + weight*fallback,
               "Saved blend is inconsistent with the validation-selected weight.")
        match_frame = frame[identity + ["international_scope"]].assign(period=period)
        matches.append(match_frame)
        global_end = splits.loc["training" if period == "validation" else "validation", "end"]
        period_metadata[period] = {"start": start.isoformat(), "end": end.isoformat(), "matches": len(frame),
                                   "global_coefficients_trained_through": global_end.isoformat(),
                                   "regional_coefficients_trained_through": regional_training_ends[period].isoformat(),
                                   "status": "development_validation" if period == "validation" else "previously_inspected_development_test"}
        saved_frames[period] = frame
    selection_end = saved_frames["validation"].loc[saved_frames["validation"]["international_scope"], "date"].max()
    if pd.isna(selection_end):
        raise ValueError("The recorded blend has no international validation cohort.")
    for period, frame in saved_frames.items():
        available = frame["regional_available"]
        columns = {"constant_50_v1": None, "global_elo_v1": "elo_probability_a",
                   "global_logistic_v1": "global_logistic_probability_a", "regional_logistic_v1": "regional_with_global_fallback",
                   "blend_v1": "blend_probability_a"}
        for model in models:
            is_regional = model.fallback_model_id is not None
            trained = None
            if model.model_id == "global_logistic_v1":
                trained = period_metadata[period]["global_coefficients_trained_through"]
            elif is_regional:
                trained = period_metadata[period]["regional_coefficients_trained_through"]
                if model.model_id == "blend_v1":
                    trained = max(trained, period_metadata[period]["global_coefficients_trained_through"])
            ledger = frame[["match_id"]].assign(
                period=period, model_id=model.model_id,
                probability_a=.5 if columns[model.model_id] is None else frame[columns[model.model_id]],
                prediction_timestamp=frame["date"], information_cutoff_exclusive=frame["date"],
                model_trained_through=trained, configuration_selected_through=selection_end if model.selection_period else None,
                native_available=available if is_regional else True, used_fallback=~available if is_regional else False,
                parameter_selection_reused=bool(model.selection_period and period == "validation"), evaluation_mode="online_elo",
            )
            ledgers.append(ledger)
    baseline = pd.read_csv(files["baseline_test"])
    baseline["date"] = utc_column(baseline["date"])
    current = saved_frames["development_test"].sort_values("match_id").reset_index(drop=True)
    baseline = baseline.sort_values("match_id").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(current[identity], baseline[identity], check_dtype=False)
    except AssertionError as exc:
        raise ValueError("Notebook 03 and 05 test populations differ.") from exc
    _close(current["elo_probability_a"], baseline["elo_probability_a"], "Global Elo predictions differ from notebook 03.")
    _close(current["global_logistic_probability_a"], baseline["logistic_probability_a"], "Global logistic predictions differ from notebook 03.")
    return pd.concat(matches, ignore_index=True), pd.concat(ledgers, ignore_index=True), models, {
        "input_sha256": hashes, "periods": period_metadata, "selection_end": selection_end.isoformat(),
        "model_metadata": metadata,
    }


def verify_existing_scores(root, metrics):
    model_mapping = {"global_elo": "global_elo_v1", "global_logistic": "global_logistic_v1",
                     "regional_logistic": "regional_logistic_v1", "validation_selected_blend": "blend_v1"}
    for filename, period in [("validation_metrics.csv", "validation"), ("test_metrics.csv", "development_test")]:
        original = pd.read_csv(Path(root) / "results/international_model" / filename)
        for row in original.itertuples(index=False):
            population = "common_native" if row.population == "common_available" else "full_scope"
            actual = metrics.loc[metrics["period"].eq(period) & metrics["scope"].eq(row.scope)
                                 & metrics["population"].eq(population) & metrics["model_id"].eq(model_mapping[row.model])]
            if len(actual) != 1:
                raise ValueError("A saved comparison row is missing.")
            _close(actual.iloc[0][["matches", "accuracy", "log_loss", "brier", "roc_auc"]].to_numpy(float),
                   [row.matches, row.accuracy, row.log_loss, row.brier, row.roc_auc], "Evaluation does not reproduce saved notebook metrics.")


def _report(tables, models, run_id):
    labels = {m.model_id: m.label for m in models}
    lines = [f"# Model evaluation: {run_id}", "", "Score-only comparison of fixed saved predictions. No training, tuning, or model selection occurred.", "",
             "Both periods are development evidence. The existing test period has already influenced project development. The blend's validation score reuses the period that selected its weight.", "",
             "## Full-scope international comparisons", "", "| Period | Model | Games | Accuracy | Log loss | Brier | ROC AUC |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    chosen = tables["metrics"].loc[tables["metrics"]["scope"].eq("international") & tables["metrics"]["population"].eq("full_scope")]
    for row in chosen.itertuples(index=False):
        lines.append(f"| {row.period} | {labels[row.model_id]} | {row.matches} | {row.accuracy:.6f} | {row.log_loss:.6f} | {row.brier:.6f} | {row.roc_auc:.6f} |")
    interval = tables["paired_differences"].iloc[0]
    lines += ["", "## Paired uncertainty for the blend versus global logistic", "",
              f"{interval['resamples']} paired week-block resamples; {interval['confidence']:.0%} percentile intervals. Resampling seed is recorded in the manifest.", "",
              "Differences are candidate minus baseline. Negative log loss/Brier and positive accuracy differences favor the candidate. These intervals condition on fixed predictions and ignore fitting/selection uncertainty.", "",
              "| Period | Scope | Metric | Difference | Interval lower | Interval upper | Weeks | Status |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    selected = tables["paired_differences"].loc[tables["paired_differences"]["model_id"].eq("blend_v1")
                                               & tables["paired_differences"]["population"].eq("full_scope")
                                               & tables["paired_differences"]["scope"].isin(["international", "worlds"])]
    for row in selected.itertuples(index=False):
        lines.append(f"| {row.period} | {row.scope} | {row.metric} | {row.candidate_minus_baseline:.6f} | {row.ci_lower:.6f} | {row.ci_upper:.6f} | {row.blocks} | {row.interval_status} |")
    lines += ["", "## Interpretation and limits", "",
              "- Full scope retains all matches with explicit fallback. Common-native scope requires native predictions from every registered model; its counts must be read separately.",
              "- The 50% baseline always predicts A at an exact 0.5 tie, matching the other models' decision rule. Its measured accuracy need not be exactly 50%.",
              "- Bootstrap units are observed UTC calendar weeks starting Monday. Whole weeks are resampled in pairs, with game-weighted means, because verified series IDs are unavailable. This is a time-dependence approximation, not a claimed series/tournament identity.",
              "- Weeks are not guaranteed independent; series/events can cross weeks. Few-week intervals are fragile. Fewer than two blocks produces no interval. Do not interpret these as definitive generalization or causal evidence.",
              "- Multiple scopes/models are descriptive comparisons, not an automated significance or winner-selection procedure. Calibration ECE depends on binning and sample size.",
              "- Timestamps represent logical pre-match information times in a retrospective backtest, not proof that these predictions were logged contemporaneously.",
              "- These are online Elo evaluations, not frozen quarterfinal forecasts or periodic-refitting backtests. Source league labels and identities remain unverified.",
              "- Reserve later untouched data for final claims. Keep frozen and online evaluations separate.", ""]
    return "\n".join(lines)


def _figures(folder, tables, models):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    labels = {m.model_id: m.label for m in models}
    chosen = tables["metrics"].loc[tables["metrics"]["period"].eq("development_test")
                                   & tables["metrics"]["scope"].eq("international") & tables["metrics"]["population"].eq("full_scope")]
    figure = Figure(figsize=(11, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for axis, metric, title in zip(axes, ["accuracy", "log_loss"], ["Accuracy (higher is better)", "Log loss (lower is better)"]):
        axis.barh([labels[x] for x in chosen["model_id"]], chosen[metric], color=["#8293a5", "#7aa6c2", "#5688ad", "#64a89d", "#c2944f"])
        axis.set_title(title)
        axis.set_xlim(0, 1 if metric == "accuracy" else max(.75, chosen[metric].max()*1.1))
        axis.grid(axis="x", alpha=.2)
    figure.suptitle("International development test — already inspected; identical matches")
    figure.savefig(folder / "metrics.png", dpi=150)
    figure = Figure(figsize=(10, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for axis, period in zip(axes, ["validation", "development_test"]):
        source = tables["calibration"].loc[tables["calibration"]["period"].eq(period) & tables["calibration"]["scope"].eq("international")
                                           & tables["calibration"]["population"].eq("full_scope") & tables["calibration"]["matches"].gt(0)]
        for model in models:
            group = source.loc[source["model_id"].eq(model.model_id)]
            axis.plot(group["predicted_probability"], group["observed_win_rate"], marker="o", markersize=3, label=model.label)
        axis.plot([0, 1], [0, 1], "--", color="gray", alpha=.6)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed A win rate", title=period.replace("_", " "))
        axis.grid(alpha=.15)
    axes[1].legend(fontsize=7, loc="lower right")
    figure.suptitle("International calibration — inspect bin counts; validation blend is selection-reused")
    figure.savefig(folder / "calibration.png", dpi=150)
    source = tables["paired_differences"].loc[tables["paired_differences"]["period"].eq("development_test")
                                              & tables["paired_differences"]["scope"].eq("international")
                                              & tables["paired_differences"]["population"].eq("full_scope")
                                              & tables["paired_differences"]["metric"].eq("log_loss")]
    figure = Figure(figsize=(9, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for index, row in enumerate(source.itertuples(index=False)):
        if np.isfinite(row.ci_lower):axis.hlines(index, row.ci_lower, row.ci_upper, color="#5688ad", linewidth=3)
        axis.scatter(row.candidate_minus_baseline, index, color="#c2944f", zorder=3)
    axis.set_yticks(range(len(source)), [labels[x] for x in source["model_id"]])
    axis.axvline(0, color="gray", linestyle="--")
    axis.set_xlabel("Candidate − global-logistic log loss (negative favors candidate)")
    axis.set_title("Paired UTC-week bootstrap: diagnostic intervals, fixed models")
    axis.grid(axis="x", alpha=.2)
    figure.savefig(folder / "paired_differences.png", dpi=150)


def run_evaluation(root=".", output=None, *, resamples=1000, seed=2026, confidence=.95, bins=10):
    root = Path(root).resolve()
    matches, predictions, models, source = load_saved_suite(root)
    configuration = {"schema_version": 1, "evaluation_mode": "online_elo", "baseline_model_id": "global_logistic_v1",
                     "bootstrap_resamples": resamples, "bootstrap_seed": seed, "confidence": confidence, "calibration_bins": bins,
                     "bootstrap_block": "UTC_calendar_week_Monday"}
    engines = {p.name: file_hash(p) for p in [Path(__file__), Path(__file__).with_name("evaluation.py")]}
    versions = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                "sklearn": sklearn.__version__, "matplotlib": version("matplotlib")}
    identity = {"inputs": source["input_sha256"], "configuration": configuration, "engine_sha256": engines, "environment": versions}
    run_id = "evaluation_v1_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    parent = Path(output) if output is not None else root / "results/evaluation"
    if not parent.is_absolute():parent = root / parent
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / run_id
    if target.exists():
        manifest_path = target / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except PermissionError as exc:
            raise ValueError(f"Cannot read {manifest_path}. Check report-folder permissions; this does not mean the run is incomplete.") from exc
        except FileNotFoundError as exc:
            raise ValueError(f"Existing run directory has no manifest: {manifest_path}. It will not be overwritten.") from exc
        if manifest["identity"] != identity:raise ValueError("Run ID collision or mismatched inputs.")
        for name, digest in manifest["output_sha256"].items():
            try:
                actual = file_hash(target / name)
            except PermissionError as exc:
                raise ValueError(f"Cannot read {target/name}. Check report-folder permissions.") from exc
            except FileNotFoundError as exc:
                raise ValueError(f"Missing existing run artifact: {target/name}. Refusing to overwrite this run.") from exc
            if actual != digest:
                raise ValueError("Existing run artifacts changed; refusing to overwrite them.")
    else:
        tables = evaluate_predictions(matches, predictions, models, resamples=resamples, seed=seed, confidence=confidence, bins=bins)
        verify_existing_scores(root, tables["metrics"])
        with report_staging(parent) as temporary:
            folder = Path(temporary) / run_id
            folder.mkdir()
            for name, frame in tables.items():frame.to_csv(folder/f"{name}.csv", index=False)
            matches.to_csv(folder/"matches.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
            predictions.to_csv(folder/"predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
            (folder/"model_registry.json").write_text(json.dumps([m.as_dict() for m in models], indent=2)+"\n", encoding="utf-8")
            (folder/"report.md").write_text(_report(tables, models, run_id), encoding="utf-8")
            _figures(folder, tables, models)
            manifest = {"run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(), "identity": identity,
                        "periods": source["periods"], "blend_selection_end": source["selection_end"],
                        "prediction_timestamp_semantics": "Logical pre-match information time; retrospective backtest, not a contemporaneous forecast log.",
                        "status": "development_only", "source_metrics_reproduced": True,
                        "validation_blend_selection_reused": True,
                        "bootstrap_limitations": "Fixed predictions; calendar-week approximation, not verified series clusters; no model-fitting or selection uncertainty.",
                        "versions": versions,
                        "output_sha256": {p.name: file_hash(p) for p in sorted(folder.iterdir())}}
            (folder/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
            folder.rename(target)
    (parent/"latest.json").write_text(json.dumps({"run_id": run_id, "directory": run_id}, indent=2)+"\n", encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    try:
        path = run_evaluation(args.root, args.output, resamples=args.bootstrap_resamples, seed=args.seed)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(f"Saved score-only development evaluation: {path}")
    print(f"Read {path/'report.md'}; this is not an untouched-final or frozen-tournament evaluation.")


if __name__ == "__main__":
    main()
