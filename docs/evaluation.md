# Model evaluation

Evaluation is separate from training and the tournament simulator. It scores fixed saved probabilities against canonical outcomes, checks fair populations and timing, and exports comparable reports. It does not fit a model, select a blend, or change the active predictor.

## Run it

After notebooks 01-05 have generated their consistent model and prediction outputs, run from the project root:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --root .
```

Or run [06_model_evaluation.ipynb](../notebooks/06_model_evaluation.ipynb), which explains the protocol and displays the reports and plots. It has eight numbered code cells.

The default protocol uses ten calibration bins, 1,000 paired bootstrap repetitions, seed 2026, and 95% percentile intervals. The CLI also accepts `--bootstrap-resamples` and `--seed`. These control diagnostic precision/reproducibility, not model selection.

The current adapter needs the canonical clean matches, global/calibrated feature exports, saved split file, model metadata, and validation/test predictions from notebooks 03/05. A fresh clone without generated analysis inputs cannot rebuild the evaluation, but can inspect the checked-in report. No joblib model is loaded or trained by the evaluator.

## What is compared

| ID | Method | Native eligibility / fallback |
| --- | --- | --- |
| `constant_50_v1` | Constant probability 0.5 | Every match; no fallback. |
| `global_elo_v1` | Raw global Elo | Every match; no fallback. |
| `global_logistic_v1` | Global Elo logistic | Every match; comparison reference. |
| `regional_logistic_v1` | Local/pool-offset logistic | Eligible international inputs; global-logistic fallback otherwise. |
| `blend_v1` | Previously validation-selected blend | Same international eligibility; global-logistic fallback otherwise. |

Every method uses the same `p >= 0.5` decision rule. The constant baseline predicts A on ties, so its accuracy equals A's observed win frequency rather than necessarily 50%. Its log loss is approximately 0.693147 and Brier score is 0.25.

The adapter verifies identities/outcomes against `matches_clean.csv`, exact notebook 03 boundaries, original feature fingerprints, training counts, native eligibility, fallback/blend arithmetic, and notebook 03's test probabilities. Newly calculated metrics must reproduce the existing notebook 05 reports.

### Populations

Each period is reported for all games, international scope, MSI plus WLDs, and WLDs, using source labels.

- **Full scope:** every scoped match is retained with explicit model fallback.
- **Common native:** matches where every registered model has a native prediction. Here that means eligible international regional inputs. It is not notebook 04's larger raw-local-Elo known-home subset.

Adding a future model can change the common-native subset. Compare full scope across runs, or explicitly fix the subset before comparing common-native results. Do not compare scores from different silently filtered samples.

## Timing and interpretation

The adapter preserves notebook 03's validation and final-period boundaries. The latter is named **`development_test`**, because its outcomes have already influenced project development. Validation is also a development period.

The blend's validation score reuses the outcomes that selected its weight. Those rows are explicitly marked `parameter_selection_reused`; the reported validation blend score is not independent generalization evidence. Actual coefficient-training endpoints and the blend-selection cohort endpoint are recorded separately.

These are retrospective **online Elo** evaluations. Earlier evaluation outcomes update ratings for later games; coefficients remain fixed during each period. Logical prediction timestamps represent pre-match information availability, not proof that forecasts were logged at those historical times.

The generic scorer rejects missing model/match rows, duplicates, invalid probabilities, mismatched forecast times, overlapping periods/equal timestamps crossing periods, future coefficient training, and mixed evaluation modes. Future configuration selection is rejected except for explicitly declared validation reuse. These checks enforce the supplied protocol; they do not independently prove source accuracy or eliminate every possible feature-generation leak.

Frozen quarterfinal forecasts and periodically retrained backtests require separately generated predictions. The saved-model adapter does not create them or infer official brackets/series identities.

## Scores and uncertainty

Reports contain accuracy, log loss, Brier score, ROC AUC, ten-bin calibration, expected calibration error (ECE), counts, and fallback coverage. Empty scopes/bins remain explicit. Single-class scopes have undefined ROC AUC. Calendar-year tables describe segments within periods, not balanced seasons or new training folds.

The paired comparison uses per-game candidate-minus-global-logistic differences in accuracy, log loss, and Brier score. Positive accuracy and negative loss differences favor the candidate. ROC AUC is scored on the complete population, not averaged across weeks.

Because the cleaned data lacks verified series IDs, bootstrap units are observed **UTC calendar weeks starting Monday**. Resample whole weeks in pairs and divide summed per-game differences by summed game counts, preserving game weighting when weeks differ in size.

This is a diagnostic time-block approximation. Weeks may remain dependent, and events can cross weeks. Fewer than two blocks yields no interval; fewer than ten is flagged as fragile. Intervals condition on fixed models/predictions and omit model-fitting and blend-selection uncertainty. Multiple model/scope comparisons are descriptive, not an automated significance or model-selection procedure.

In the current international development-test report, the blend's log loss is lower than global logistic's, but the paired interval spans zero. This means the point improvement alone is not strong final evidence. The Worlds subset has only nine observed week blocks, so its interval needs particular caution. Reserve later untouched data for final claims.

## Versioned outputs

Generated reports remain local and are excluded from Git. After running evaluation, `results/evaluation/latest.json` points to the current `evaluation_v1_<hash>/` folder. The identifier includes input fingerprints, engine source, configuration, and library versions. Identical reruns reuse the same immutable files. Changed inputs/code/settings/environment produce a new run. Changed or incomplete existing outputs are rejected rather than overwritten.

| Artifact | Purpose |
| --- | --- |
| `report.md` | Human-readable comparison and limitations. |
| `metrics.csv` | All period/scope/population/model scores. |
| `coverage.csv` | Native/fallback counts and pre-filter scope sizes. |
| `calibration.csv` | Fixed-bin predictions, observed rates, and counts. |
| `paired_differences.csv` | Paired point differences, bootstrap intervals, block counts, and warnings. |
| `temporal_metrics.csv` | Calendar-year segments and coverage. |
| `matches.csv.gz` | Canonical identities, dates, and outcomes, separate from predictions. |
| `predictions.csv.gz` | Normalized model/match ledger with probabilities, timing, availability, and selection-reuse labels. |
| `model_registry.json` | Model IDs, versions, features, settings, selection period, and fallback dependencies. |
| `manifest.json` | Source fingerprints, timing/status, configuration, environment, and output hashes. |
| `metrics.png`, `calibration.png`, `paired_differences.png` | Exportable comparison figures. |

CSV ledgers are gzip-compressed to avoid large repeated plain-text files; pandas can read them directly. This schema can support a later reporting dashboard without implementing evaluation logic again in the dashboard.

On Windows, staging and published reports inherit their output parent's normal permissions. An earlier staging implementation could publish private temporary-directory permissions when run under a sandbox account. That could make an existing manifest unreadable to the regular user and misleadingly look like an incomplete run. The staging implementation is fixed, and read-permission failures are now reported separately from missing manifests. No administrator terminal is required for ordinary evaluation runs in a writable project folder.

## Add a future model version

Train/select the new model outside the evaluator. Give it a new `ModelSpec.model_id` and version, then export a complete ledger for the same canonical matches. Supply fallback predictions for unavailable native features rather than dropping rows.

`src/evaluation.py` provides `ModelSpec`, `validate_predictions`, and `evaluate_predictions`. The saved 03/05 adapter and versioned writer live in `src/evaluate.py`; extend the adapter with the new source file and provenance when adding another model to the CLI suite. Metric and resampling definitions do not need to change.

The ledger contract includes `match_id`, `period`, `model_id`, `probability_a`, logical `prediction_timestamp`, `information_cutoff_exclusive`, `model_trained_through`, `configuration_selected_through`, `native_available`, `used_fallback`, `parameter_selection_reused`, and `evaluation_mode`. Keep outcomes in the canonical match table. All models must use identical information and forecast times on each compared match.

For an untouched future period, freeze the model/features/selection protocol before inspecting outcomes. The existing development period cannot become untouched again.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Evaluation tests cover known metric values, tie rules, bin boundaries, exact/empty probabilities, complete populations, timestamp and mode consistency, explicit selection reuse, fallback arithmetic, paired block weighting, reproducibility, insufficient blocks, input nonmutation, and immutable run reuse/tamper detection. Existing predictor/simulator tests remain part of the suite.
