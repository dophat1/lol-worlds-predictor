# LoL Worlds Predictor

A notebook-based project for estimating League of Legends game win probabilities from chronological match history. It combines global Elo, domestic-pool Elo, international pool-strength calibration, and logistic regression, with an initial Monte Carlo series example.

The current implementation evaluates individual games as history unfolds. A complete tournament forecasting system, historical gold/objective features, periodic model retraining, and resumable state are still planned.

**Evaluation status:** the saved test period has already been inspected and has influenced development. Its results are development evidence, not an untouched final benchmark or a forecast frozen before a tournament.

## Contents

- [What is implemented](#what-is-implemented)
- [Setup and execution](#setup-and-execution)
- [Data and cleaning](#data-and-cleaning)
- [How the models work](#how-the-models-work)
- [Chronological evaluation](#chronological-evaluation)
- [Saved results](#saved-results)
- [Repository and generated artifacts](#repository-and-generated-artifacts)
- [Validation and reproducibility](#validation-and-reproducibility)
- [Limitations and next steps](#limitations-and-next-steps)

## What is implemented

Each notebook contains an overview and numbered explanations before its working code cells.

| Notebook | Purpose | Main outputs |
| --- | --- | --- |
| [01 — Data exploration](notebooks/01_data_exploration.ipynb) | Read every available raw CSV, select team rows, validate games, and preserve statistics separately. | Clean matches, team statistics, cleaning audits. |
| [02 — Global Elo](notebooks/02_elo_feature_engineering.ipynb) | Replay all accepted games with ratings carried across years and predictions recorded before updates. | Global pre-match features and final ratings. |
| [03 — Baseline model](notebooks/03_baseline_model.ipynb) | Evaluate global Elo and an Elo-only logistic model using chronological splits. | Baseline model, exact split boundaries, predictions, metrics, calibration. |
| [04 — International evaluation](notebooks/04_international_evaluation.ipynb) | Classify competitions, track historical domestic homes and local Elo, learn pool offsets, and compare probabilities. | Classified, regional, and calibrated histories; regional evaluation reports. |
| [05 — Simulation illustration](notebooks/05_worlds_simulation.ipynb) | Inspect retrospective source-labeled games and demonstrate game/best-of-five Monte Carlo mechanics. | Notebook tables and illustrative simulation results. |
| [06 — International model](notebooks/06_international_model.ipynb) | Fit separate local-strength and pool-strength coefficients, select a blend on validation, and refit for evaluation. | Model bundle, selection record, predictions, metrics, coverage, calibration. |

The modeling path is **01 → 02 → 03 → 04 → 06**. Notebook 05 needs notebook 02 and is an independent learning exercise; notebook 06 does not consume its simulations.

## Setup and execution

### 1. Create a Python environment

From the repository root, using Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\jupyter-notebook.exe
```

Select the virtual environment's Python kernel in Jupyter. The notebooks support starting from the project root or the `notebooks/` directory.

The local environment inspected for this README uses Python **3.14.5**, pandas **3.1.0**, NumPy **2.5.4**, and scikit-learn **1.9.1**. Dependencies in [requirements.txt](requirements.txt) are currently unpinned; this is not a locked environment or a guarantee of identical results with other versions. The notebooks also use joblib for model serialization, installed as a scikit-learn dependency.

### 2. Supply the raw data

Place the source Oracle's Elixir CSV files directly in `data/raw/`. The cleaning notebook reads every `*.csv` in that folder, rather than hard-coding one season. There is no automatic downloader.

The saved all-year run used ten files named `2017_LoL_esports_match_data_from_OraclesElixir.csv` through `2026_LoL_esports_match_data_from_OraclesElixir.csv`. **Those ten files are not all distributed in Git:** a legacy 2024 CSV remains tracked, while the raw-data directory is otherwise ignored. A fresh clone alone therefore cannot reproduce the all-year results. Obtain the missing source files separately and check their coverage against the input-file audit.

### 3. Run the notebooks in order

Use **Restart Kernel and Run All Cells** for each notebook, following the modeling path above. Notebook 05 can run after 02, or between 04 and 06 if following filename order.

An optional headless run of all six notebooks, from the repository root:

```powershell
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
Get-ChildItem .\notebooks\*.ipynb | Sort-Object Name | ForEach-Object {
    & .\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=900 $_.FullName
    if ($LASTEXITCODE -ne 0) { throw "Notebook failed: $($_.Name)" }
}
```

Execution regenerates processed CSVs, models, reports, and notebook outputs. Keep any snapshot you want to compare before rerunning. Changing raw files can also change the 60%/80% timestamp boundaries; downstream notebooks reuse the boundaries from that particular baseline run.

## Data and cleaning

### Input fields and prediction unit

Required source columns are `gameid`, `date`, `league`, `teamname`, `position`, and `result`. Only `position == "team"` rows are retained. Available optional columns include team ID, side, patch, gold difference at 15 minutes, dragons, towers, barons, and corresponding opponent objectives.

The compact match-history table has six columns:

| Column | Meaning |
| --- | --- |
| `match_id` | Composite of UTC date, league, and source game ID. |
| `date` | Parsed UTC timestamp used for ordering and batching. |
| `league` | Competition label from the source data. |
| `team_a`, `team_b` | Opponents assigned alphabetically within a game. A/B does not encode map side. |
| `result_a` | 1 if A won, otherwise 0. |

### Acceptance rules and audit limits

Notebook 01 normalizes identity text and dates, separates missing game identities, and removes exact duplicates across the selected fields except source filename. Each accepted composite game group must contain:

- Exactly two team records and two distinct team names.
- Complete required fields and binary outcomes.
- Exactly one winner.

Rejected groups and underlying rows remain available in audit exports. The notebook separately reports source game IDs appearing under different dates or leagues. **A composite key does not resolve inconsistent metadata:** the reuse audit reports these contexts but does not automatically adjudicate every conflict. Review a nonempty audit before relying on the resulting history.

Statistics are saved in `team_game_statistics_clean.csv`, separately from `matches_clean.csv`. Unknown statistics are not filled with zero. These are same-game statistics; using them directly to predict that game's result would leak information. Historical aggregates using only earlier games have not yet been implemented.

### Saved cleaning snapshot

The latest local cleaning summary records:

| Item | Value |
| --- | ---: |
| Input CSV files | 10 |
| Original team rows | 192,182 |
| Accepted games | 96,075 |
| Accepted team-statistics rows | 192,150 |
| Rejected game groups | 16 |
| Exact duplicate rows removed | 0 |
| Source game IDs with multiple date/league contexts | 0 |

Accepted dates span **2017-01-10 15:44:06 UTC** through **2026-10-26 21:54:37 UTC**. These counts describe the supplied local files. Source competition labels and their identities have not been independently verified.

## How the models work

### Global Elo

Global Elo places every exact team-name identity on one rating scale. A new team starts at 1500; ratings persist across years without a season reset or inactivity decay.

```text
p(A wins) = 1 / (1 + 10 ** ((rating_B - rating_A) / 400))
delta     = 24 * (result_A - p(A wins))
rating_A  = rating_A + delta
rating_B  = rating_B - delta
```

An unexpected win transfers more points than an expected win. Equal-and-opposite transfers conserve rating. The feature table records `pre_elo_a`, `pre_elo_b`, `elo_diff`, and `elo_probability_a` before the game's result updates state.

### Competition roles and domestic pools

Regional learning uses explicit competition metadata:

| Field | Purpose |
| --- | --- |
| `competition_type` | Domestic, international, cross-pool, developmental, or review. |
| `rating_pool` | Domestic pool assigned to eligible domestic participation. |
| `updates_home_pool` | Whether participation establishes the team's domestic home. |
| `use_local_elo` | Whether the game updates local ratings. |
| `use_region_calibration` | Whether the game may update pool offsets, subject to further eligibility checks. |
| `use_international_model` | Whether the game belongs to the international logistic-model scope. |

Domestic pools are `LCK`, `LPL`, `LEC`, `LCS`, `PCS`, `VCS`, `CBLOL`, `LLA`, `LJL`, `TCL`, `LCO`, `LMS`, `OPL`, `LCL`, `LCP`, `LTA_N`, and `LTA_S`. Source labels `LTA N` and `LTA S` map to the last two pools. `EU LCS → LEC` and `NA LCS → LCS` are explicit continuity assumptions; other histories are not automatically merged.

The initial international modeling scope is `MSI`, `WLDs`, and `FST`. `LTA` is a cross-pool bridge: eligible games can calibrate existing pools, but participation does not create a new domestic home. Listed developmental competitions do not inherit a parent pool. Unreviewed competitions remain in global Elo but do not update regional state.

The mapping is currently league-wide, not year-specific. Renamed or restructured competitions can have different historical roles, so this mapping needs further review.

### Local Elo and historical homes

Local Elo is keyed by **`(rating_pool, team)`**, with initial rating 1500, K=24, and scale=400. Only eligible domestic games update it; international results do not.

A domestic game's competition identifies the applicable pool for that game. At an international game, a team's home comes only from domestic appearances at **earlier timestamps**. Missing homes remain missing. Conflicting domestic assignments for a team at one timestamp are rejected. Final home summaries are never joined backward onto historical rows.

Local ratings measure strength relative to a pool. A strong local record alone does not show how that pool compares with another one.

### International pool-strength calibration

Pool offsets connect those otherwise separate local scales:

```text
adjusted_strength = pre_match_local_elo + pre_match_pool_offset
p(A wins)         = Elo probability from the two adjusted strengths
delta_pool        = 12 * (result_A - p(A wins))
offset_pool_A    += delta_pool
offset_pool_B    -= delta_pool
```

Offsets start at zero. Region K=12 and scale=400 are fixed baseline settings, not settings selected using test performance. Update only after a flagged game between distinct known pools with finite local ratings. Unknown homes and same-pool games cannot change offsets.

Store prior offsets, adjusted strengths and differences, probabilities, eligibility, transfers, and prior calibration-game counts. Each eligible game increments both participating pools' counts. The saved full history retains all 96,075 games and contains 1,645 eligible calibration games.

### Logistic models and validation-selected blending

The global logistic model learns a probability transformation of `elo_diff`. The international model uses two features, `local_elo_diff` and `pool_offset_diff`, so it can learn separate coefficients for local performance and relative pool strength.

Both use `StandardScaler(with_mean=False)` fitted inside the training pipeline, logistic regression with **C=1**, no intercept, and mirrored examples: each `(x, result)` is paired with `(-x, 1 - result)`. This preserves opponent symmetry; a zero feature vector gives probability 0.5.

The global model trains on all development games. The international model trains only on modeling-scope games with available regional features. Notebook 06 evaluates the predeclared blend weights 0, 0.1, ..., 1 on full-scope international **validation log loss**, preferring the smaller regional weight on ties:

```text
p_blend = (1 - regional_weight) * p_global_logistic
          + regional_weight * p_regional_logistic
```

The saved validation run selected **0.8**, using 423 international validation games. The regional model used 1,047 games during initial training and 1,470 after refitting on training plus validation. Feature definitions, regularization, and selected weight remain fixed for the final-period evaluation.

Fallback policies differ deliberately:

- **Notebook 04:** local and calibrated Elo use **global Elo** when regional strengths are unavailable. Regional predictions can be compared wherever homes and local ratings are known.
- **Notebook 06:** regional logistic and its blend use **global logistic** outside `MSI`/`WLDs`/`FST` or when regional features are unavailable. The international model is not applied to domestic games.

### Monte Carlo illustration

Notebook 05 assumes an illustrative single-game probability of **0.3257**, independent games, and a best-of-five ending at three wins. It uses seed 2026 and 10,000 simulations for each example, then checks simulated frequencies against exact game and binomial series probabilities.

That probability is not a fitted matchup estimate. The notebook's source-labeled retrospective ratings and its simulation exercise do not supply an official bracket, official participants, or a tournament forecast.

## Chronological evaluation

Notebook 03 places split boundaries at 60% and 80% of **distinct timestamps**. Equal timestamps never cross periods. Notebooks 04 and 06 reuse its [saved boundaries](results/baseline/chronological_splits.csv) and verify match identity alignment.

| Period | Games | Start, UTC | End, UTC |
| --- | ---: | --- | --- |
| Training | 57,656 | 2017-01-10 15:44:06 | 2023-03-04 22:15:44 |
| Validation | 19,215 | 2023-03-04 22:51:24 | 2025-02-05 18:49:52 |
| Test | 19,204 | 2025-02-05 18:58:03 | 2026-10-26 21:54:37 |

For global ratings, local ratings, home membership, offsets, and their prior counts, each timestamp batch reads frozen state before summed changes are applied. Later outcomes cannot affect earlier features. Logistic scaling and coefficients are fitted only on the applicable training period, then remain fixed while scoring its evaluation period.

This is **online, walk-forward Elo evaluation**: earlier validation or test results can update state before later evaluation games. A forecast frozen before a tournament would instead use only information available at that earlier cutoff. Periodically retraining logistic coefficients during evaluation is not implemented yet.

Metrics include accuracy at a 0.5 threshold, log loss, Brier score, ROC AUC, and ten fixed-width calibration bins. Lower log loss/Brier and higher accuracy/AUC are preferable. Accuracy alone does not establish good probabilities; inspect confidence, calibration-bin counts, and scope coverage too.

## Saved results

The following numbers are rounded from checked-in CSV reports, not newly invented or simulated evaluation results. They refer to the saved input snapshot and split above. Small international samples and prior test inspection limit conclusions.

### Global baseline: all 19,204 test games

| Model | Accuracy | Log loss | Brier | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| Global Elo | 0.630077 | 0.640541 | 0.224841 | 0.681242 |
| Global Elo logistic | 0.630077 | 0.640172 | 0.224736 | 0.681242 |

The logistic baseline changes confidence slightly without changing accuracy. Its log-loss improvement is small. Source: [baseline scope metrics](results/baseline/test_metrics_by_scope.csv).

### Raw regional Elo: common subset and full scope

The common known-home subset contains 7,431 games. Full scope retains all 19,204, with global-Elo fallback on 11,773 games. This is **38.70% regional availability**, not an evaluation that silently discards the remaining games.

| Method | Common-subset accuracy | Common-subset log loss | Full-scope accuracy | Full-scope log loss |
| --- | ---: | ---: | ---: | ---: |
| Global Elo | 0.629660 | 0.645343 | 0.630077 | 0.640541 |
| Local-only Elo | 0.630332 | 0.644119 | 0.630337 | 0.640067 |
| Calibrated Elo | 0.635042 | 0.641827 | 0.632160 | 0.639181 |

Source: [regional metrics](results/regional/test_metrics_by_scope.csv) and [coverage](results/regional/test_coverage.csv). International test coverage is 390/390; MSI plus WLDs is 310/310; WLDs is 159/159. Common-subset and full-scope populations therefore coincide within those saved international scopes.

### Source-labeled WLDs: the same 159 test games

| Model | Accuracy | Log loss | Brier | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| Global Elo | 0.578616 | 0.673760 | 0.240903 | 0.603180 |
| Global Elo logistic | 0.578616 | 0.673131 | 0.240555 | 0.603180 |
| Local-only Elo | 0.603774 | 0.693585 | 0.248665 | 0.592369 |
| Calibrated Elo | 0.622642 | 0.636894 | 0.224327 | 0.686328 |
| International regional logistic | 0.666667 | 0.626133 | 0.218259 | 0.706359 |
| Validation-selected logistic blend | 0.672956 | 0.624208 | 0.218018 | 0.705087 |

All 159 games have regional features. The raw Elo rows come from the regional report; logistic rows come from [international-model test metrics](results/international_model/test_metrics.csv).

Across all **390 international test games**, blend log loss is **0.634704**, versus **0.652942** for global logistic. Regional logistic alone has log loss **0.654603**, slightly worse than global logistic despite higher accuracy. Improvement in one scope or one metric does not establish universal superiority. Across all 19,204 games, the blend's log loss is **0.639801**; it uses global logistic outside the 390 international games.

See the [validation blend search](results/international_model/validation_blend_grid.csv), [model metadata](results/international_model/model_metadata.json), and calibration CSVs in the corresponding result directories. No test-result search selected the blend, but the test period itself is already part of the development feedback loop. Reserve a later untouched period before making final generalization claims.

## Repository and generated artifacts

```text
notebooks/                       Six explained, executed notebooks
results/
  baseline/                      Global Elo/logistic evaluation and splits
  regional/                      Raw local/calibrated Elo comparisons
  international_model/           Logistic/blend selection and evaluation
  execution/                     Recorded notebook execution summary
src/__init__.py                  Placeholder for future reusable modules
requirements.txt                 Current, unpinned Python dependencies
data/raw/                        Source CSVs; mostly ignored by Git
data/processed/                  Generated histories and audits; ignored
models/                          Generated model bundles; ignored
```

Important generated outputs:

| Producer | Files |
| --- | --- |
| 01 | `matches_clean.csv`, `team_game_statistics_clean.csv`, `cleaning_summary.csv`, `input_file_audit.csv`, duplicate/rejection/missing-identity/game-ID-context audits. |
| 02 | `elo_features_all_years.csv`, `elo_ratings_all_years.csv`. |
| 03 | `models/elo_logistic_baseline.joblib`; reports under `results/baseline/`. |
| 04 | `competition_classification.csv`, `competitions_to_review.csv`, `matches_classified.csv`, `regional_elo_features_all_years.csv`, `local_elo_ratings_all_years.csv`, `latest_team_home_pools.csv`, `international_unknown_home_matches.csv`, `calibrated_elo_features_all_years.csv`, `regional_calibration_summary.csv`; reports under `results/regional/`. |
| 06 | `models/international_logistic_blend.joblib`; reports and metadata under `results/international_model/`. |

CSV filenames without a directory in this table are under `data/processed/`. Older exploratory processed CSVs or model files may also exist locally; the current notebooks use the canonical filenames listed here. End-of-history rating and home summaries are for inspecting final state, not for constructing historical features.

## Validation and reproducibility

The [recorded execution](results/execution/notebook_run.json) reports successful execution of **84 working code cells across six notebooks**. Subsequent Markdown explanations preserved code, saved outputs, and execution counts. Ordinary manual or headless notebook execution does not automatically refresh this separate execution-summary file.

Validation currently lives in notebook assertions and labeled synthetic fixtures. There is no populated standalone test suite or packaged production predictor yet. Checks include:

- Required fields, binary outcomes, distinct opponents, one winner, and duplicate identities.
- Elo probability symmetry, initial ratings, rating conservation, and carryover across years.
- Equal-timestamp isolation, pre-match independence from current outcomes, historical home membership, and conflicting simultaneous domestic assignments.
- Zero-sum pool transfers and excluded updates for unknown homes, same-pool games, disabled flags, and nonfinite strengths.
- Exact evaluation population alignment, fallback coverage, baseline metric reproduction, and persisted global-model prediction equivalence.
- Monte Carlo frequencies checked against exact illustrative probabilities.

International model metadata records input-file SHA-256 hashes, scikit-learn version, feature names, split dates, training counts, fixed settings, and blend selection. These improve traceability but do not replace a full dependency lock or source-data archive. Updating data or dependencies can change the saved snapshot, and exact checkpoint-resume equivalence has not been implemented.

## Limitations and next steps

Current limitations include exact-name identities without rename/roster handling, homes that can become stale, year-independent competition roles, unreviewed competitions, unequal source coverage, and small international evaluation samples. Cleaning audits identify several classes of problems but are not a complete correction-handling system.

The next milestones are:

1. Review competition roles by year and strengthen team-identity and home-membership handling.
2. Build historical gold/objective features using only earlier games, with explicit missingness and availability counts.
3. Extract reusable state-update code and add periodic chronological retraining of logistic models.
4. Implement checkpoint/resume equivalence and explicit duplicate/correction handling.
5. Add cutoff-based, frozen pre-tournament forecasts, keeping them distinct from online evaluation.
6. Expand series and tournament Monte Carlo using a supplied bracket and rules, or clearly labeled assumptions.

The project currently provides an explained research workflow and saved development evidence. Official participants, competition identities, tournament rules, and final forecasting performance must not be inferred from the illustrative simulation or invented to fill missing inputs.
