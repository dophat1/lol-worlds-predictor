# Standalone predictor v1

The analysis notebooks prepare the data and fit the models. A portable artifact contains fixed model coefficients plus rating state before an exclusive cutoff. The same prediction code serves the tournament simulator and can now be queried independently of its browser interface.

## Query a game and best-of-five

From the repository root, after exporting the model/state package:

```powershell
.\.venv\Scripts\python.exe -m src.predict_match --artifact models/simulator_artifact.json --team-a "Gen.G" --team-b "T1"
```

This is an illustrative matchup query using identities in the current dataset, not an asserted actual fixture. Replace the team names with exact identities present in your artifact.

To condition the series calculation on an assumed 2-1 score:

```powershell
.\.venv\Scripts\python.exe -m src.predict_match --artifact models/simulator_artifact.json --team-a "Gen.G" --team-b "T1" --wins-a 2 --wins-b 1
```

The query never records results or updates ratings. It uses frozen snapshot probabilities for the remaining games. Use the simulator's actual-game entry to record real outcomes and update state.

For a game outside the international-model scope, explicitly select global logistic only:

```powershell
.\.venv\Scripts\python.exe -m src.predict_match --artifact models/simulator_artifact.json --team-a "Gen.G" --team-b "T1" --scope global
```

The initial international model was trained on eligible `MSI`, `WLDs`, and `FST` games. Do not automatically apply that model to unrelated domestic or developmental competitions.

## Read the output

The JSON report includes:

- Global Elo, known home pool, local Elo, pool offset, adjusted strength, and available prior-game counts for both teams.
- Global, regional, blended, and complementary game probabilities.
- Whether fallback was used, its reason, and the effective regional weight.
- Exact conditional best-of-five probabilities and the supplied score.
- Model training endpoint, snapshot cutoff, last source-game timestamp, and development-evaluation status.

Unknown homes or local ratings remain `null`; they are not represented as zero strength. Global-only scope and missing regional inputs both use global logistic, with distinct reasons. Missing team identities, self-matches, invalid scores, and invalid model-training cutoffs are rejected.

## V1 validation

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The current suite contains 22 mechanics, prediction-contract, and HTTP/session tests. The standalone prediction checks cover symmetry, direct logistic/blend arithmetic, global-only eligibility, missing-feature fallback, score conditioning without state mutation, terminal/invalid scores, serialization, and invalid identities/cutoffs.

The [saved inference verification](../results/predictor/inference_verification.json) compares 144 matchup queries in both international and global-only modes against the original saved scikit-learn pipelines. Both known-region and fallback branches were exercised; the maximum absolute difference was approximately `2.22e-16`. These are mathematical equivalence checks, not actual match records or new predictive-performance results.

## What “v1” means

V1 provides usable game/series inference and a local fixed-bracket simulator with auditable cutoffs and actual-result replay. It is not a declaration that every research milestone is complete. Historical gold/objective features, periodic logistic retraining, robust year-specific competition/identity review, untouched tournament backtesting, and public deployment remain unfinished.

The current exported model was fitted through February 5, 2025; its state package includes source games through October 26, 2026. This reflects the fixed-coefficient evaluation design. Choosing a newer state cutoff does not retrain the coefficients or turn this package into a historical pre-quarterfinal forecast.
