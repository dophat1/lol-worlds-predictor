# LoL Worlds Predictor

A rating-based League of Legends predictor and local quarterfinal-to-title simulator. Notebooks prepare data and evaluate models; Python modules serve predictions and simulate brackets.

**Current scope:** global Elo, domestic local Elo, international pool-strength offsets, logistic regression, and best-of-five Monte Carlo. Gold/objective statistics are preserved but are **not model inputs yet**. This is a research prototype, with development results rather than an untouched final benchmark.

## Start here

| Goal | Guide |
| --- | --- |
| Run the real-data tournament simulator | [Simulator instructions](docs/simulator.md) |
| Query a game or series without the browser | [Standalone predictor](docs/predictor.md) |
| Inspect model selection and results | [Model metadata](results/international_model/model_metadata.json), [validation search](results/international_model/validation_blend_grid.csv), [test metrics](results/international_model/test_metrics.csv) |

## Setup

From the repository root in Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements are currently unpinned. The inspected local environment uses Python 3.14.5, pandas 3.1.0, NumPy 2.5.4, and scikit-learn 1.9.1; other environments have not been validated for identical results.

### Prepare the real data and model

Place Oracle's Elixir CSVs directly in `data/raw/`, then start Jupyter:

```powershell
.\.venv\Scripts\jupyter-notebook.exe
```

Run the active analysis notebooks top to bottom in this order:

| Step | Notebook | Responsibility |
| --- | --- | --- |
| 01 | [Data exploration](notebooks/01_data_exploration.ipynb) | Read every available CSV, select team rows, validate games, preserve statistics, and export audits. |
| 02 | [Global Elo](notebooks/02_elo_feature_engineering.ipynb) | Record pre-match ratings/probabilities and update chronological timestamp batches. |
| 03 | [Baseline model](notebooks/03_baseline_model.ipynb) | Define timestamp splits and evaluate global Elo against Elo-only logistic regression. |
| 04 | [International evaluation](notebooks/04_international_evaluation.ipynb) | Classify competitions, track domestic homes/local Elo, learn pool offsets, and compare aligned predictions. |
| 05 | [International model](notebooks/05_international_model.ipynb) | Fit regional logistic regression, select a blend on validation, and save the final model/evaluation. |

Run **01 -> 02 -> 03 -> 04 -> 05**. Prediction and tournament simulation live in `src/`, keeping the notebooks focused on analysis and preparation.

**A fresh clone does not contain the full dataset or generated models.** The saved all-year run used ten CSVs covering 2017-2026; only a legacy 2024 raw CSV is tracked. Obtain the missing inputs separately. There is no downloader. Rerunning notebooks overwrites generated artifacts and may change split boundaries if inputs change.

### Run the trained simulator

After generating the analysis outputs, export a snapshot and start the app:

```powershell
.\.venv\Scripts\python.exe -m src.export_artifact --cutoff 2026-10-27T00:00:00Z --output models/simulator_artifact.json
.\.venv\Scripts\python.exe -m src.simulator_app --artifact models/simulator_artifact.json --session data/processed/simulator_sessions/real_data.json --port 8501
```

Open **http://127.0.0.1:8501**. Supply eight teams and four quarterfinal pairings, then simulate. Enter actual game winners to update scores and probabilities. Sessions support download/resume and undo.

The example cutoff is just after the last saved source game, **not an asserted historical quarterfinal date**. A historical forecast needs the actual bracket and a pre-event cutoff, with model training completed earlier. The exporter rejects a model trained at/after its cutoff. A local starter session may contain an explicitly assumed what-if bracket; it is not an official bracket and is not distributed in Git.

For a clearly synthetic demonstration without a trained artifact, use `python -m src.simulator_app --demo` in the project environment. Do not run two servers on the same port. Public hosting is not implemented.

### Query a matchup

```powershell
.\.venv\Scripts\python.exe -m src.predict_match --artifact models/simulator_artifact.json --team-a "Gen.G" --team-b "T1"
```

This is an illustrative query, not an asserted actual fixture. The report includes game/series probabilities, feature values, cutoffs, and fallback reasons. `--wins-a 2 --wins-b 1` conditions the series on a score without recording results; `--scope global` uses global logistic outside international scope.

## How predictions work

| Component | Inputs and update rule |
| --- | --- |
| Global Elo | All earlier accepted games; new teams start at 1500, K=24, scale=400; ratings carry across years. |
| Local Elo | `(rating_pool, team)` ratings; eligible domestic games only; initial 1500, K=24, scale=400. |
| Pool offsets | Initially zero; eligible cross-pool games transfer equal/opposite offsets with K=12 and scale=400. |
| Global logistic | One feature: global Elo difference. |
| International logistic | Two features: local Elo difference and pool-offset difference. |
| Blend | Validation selected 80% international logistic + 20% global logistic. |

Elo probability is `1 / (1 + 10 ** ((rating_b - rating_a) / 400))`. Updates use `K * (result - probability)`. Regional adjusted strength is local Elo plus its pool offset.

International homes come only from earlier domestic appearances. Unknown homes stay missing. All equal-timestamp predictions read frozen state before summed updates; final ratings/homes are never joined backward onto history.

Both logistic pipelines fit a scaler on mirrored training examples, use C=1 and no intercept, and preserve opponent symmetry. Initial international scope is source labels `MSI`, `WLDs`, and `FST`. Roles and domestic-pool assumptions are documented in notebook 04.

Fallback is explicit: notebook 04's raw local/calibrated comparisons use global **Elo** when regional inputs are missing. The runtime international logistic model uses global **logistic** when inputs are unavailable or outside its scope.

The simulator uses a fixed eight-team bracket, with QF1/QF2 feeding SF1 and QF3/QF4 feeding SF2. Every series is best-of-five. Simulations freeze matchup probabilities; only actual games update global Elo and eligible pool offsets. Logistic coefficients and local ratings remain fixed during those international updates.

## Data and evaluation

The saved cleaning snapshot contains **96,075 accepted games** and **192,150 team-statistics rows**, spanning January 10, 2017 to October 26, 2026. Sixteen game groups were rejected. Required fields, two distinct opponents, binary results, one winner, duplicates, and game-ID context reuse are audited. A composite key does not resolve conflicting metadata; nonempty reuse audits require review. Source competition identities are not independently verified.

The six-column match history is `match_id, date, league, team_a, team_b, result_a`. A/B is alphabetical, not map side. Optional gold/objective statistics remain separate and missing values are not filled with zero.

[Saved timestamp boundaries](results/baseline/chronological_splits.csv) use 60%/80% of distinct timestamps; simultaneous games never cross periods:

| Period | Games | Start, UTC | End, UTC |
| --- | ---: | --- | --- |
| Training | 57,656 | 2017-01-10 15:44:06 | 2023-03-04 22:15:44 |
| Validation | 19,215 | 2023-03-04 22:51:24 | 2025-02-05 18:49:52 |
| Test | 19,204 | 2025-02-05 18:58:03 | 2026-10-26 21:54:37 |

Earlier evaluation outcomes update Elo for later games, while logistic coefficients stay fixed. These **online evaluation** results are distinct from a forecast frozen before a tournament.

Selected saved development results on identical matches:

| Scope | Games | Global-logistic log loss | Blend log loss | Blend accuracy |
| --- | ---: | ---: | ---: | ---: |
| All | 19,204 | 0.640172 | 0.639801 | 0.631223 |
| International | 390 | 0.652942 | 0.634704 | 0.666667 |
| Source-labeled WLDs | 159 | 0.673131 | 0.624208 | 0.672956 |

Lower log loss is better. These are game-level scores, not tournament-winner accuracy. The global-only logistic gain over raw global Elo is small (all-game log loss 0.640541 to 0.640172). International samples are limited and the test period has influenced development; reserve a later untouched period for final claims.

[Regional metrics](results/regional/test_metrics_by_scope.csv) report both common-subset and full-scope comparisons. Raw regional features cover 7,431/19,204 test games (38.70%); full scope uses global-Elo fallback for 11,773 games. Notebook 05 applies its regional model only to the 390 international games. See [coverage](results/regional/test_coverage.csv), [blend selection](results/international_model/validation_blend_grid.csv), and calibration CSVs for details.

## Validation and project layout

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The current 22 tests cover probability arithmetic, symmetry, scope/fallback, cutoff rejection, timestamp isolation, series conditioning, fixed advancement, reproducibility, and actual-result replay/save/undo. [Inference verification](results/predictor/inference_verification.json) checks 144 matchup queries in both scopes against the saved scikit-learn pipelines. Passing mechanics tests is not proof of forecasting accuracy.

```text
notebooks/             Analysis and preparation (01 through 05)
src/                   Prediction, state export, tournament engine, local app
web/                   Browser interface
tests/                 Automated tests
docs/                  Public predictor and simulator usage guides
results/               Saved predictions, metrics, calibration, metadata
data/raw/              Source CSVs; mostly ignored
data/processed/        Generated histories/audits/sessions; ignored
models/                Generated model and state packages; ignored
```

The [execution record](results/execution/notebook_run.json) lists the five current notebooks and their 69 working code cells. The obsolete simulation illustration and legacy generated artifacts have been removed. Source data, current models, and saved evaluation results remain available.

## Remaining work

Year-specific competition roles, rename/roster handling, stale homes, unequal data coverage, and unreviewed labels remain limitations. The current package's logistic model was fitted through February 5, 2025; its latest rating snapshot includes October 26, 2026. A newer state cutoff does not retrain coefficients.

Next: verified frozen tournament backtests, historical gold/objective features, controlled periodic retraining, general ingestion/correction checkpoints, and public deployment with browser verification. Power BI is deferred. Dependencies are unpinned and full source inputs are not distributed in Git, so exact reproduction needs the matching data and environment.
