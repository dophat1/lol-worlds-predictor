# Quarterfinal-to-title simulator

Notebooks remain the analysis and data-preparation workflow. The simulator is separate Python code plus a local browser interface, using an exported model and a versioned rating-state snapshot.

## Try the hypothetical demonstration

From the project root, after installing `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe -m src.simulator_app --demo
```

Open **http://127.0.0.1:8501**. No additional Python dependencies are required beyond the existing environment. Stop the server with Ctrl+C.

The demo uses eight explicitly synthetic, equal-strength teams and illustrative Elo probabilities. It is not the trained model and makes no claim about actual participants or matches. Its fixed cutoff is also illustrative. The initial title chances should be near 12.5% per team, with sampling variation; this is a symmetry check, not an evaluation result.

## Use the trained model

First run notebooks 01-04 and 06 to generate consistent model and feature files. These generated files are ignored by Git, so a fresh clone needs the source data and notebook outputs before trained-model export can run.

Export state using a timezone-aware **exclusive cutoff**:

```powershell
.\.venv\Scripts\python.exe -m src.export_artifact --cutoff 2026-10-27T00:00:00Z --output models/simulator_artifact.json
.\.venv\Scripts\python.exe -m src.simulator_app --artifact models/simulator_artifact.json
```

That example cutoff is just after the last game in the current local data. It demonstrates exporting available state; it is **not an asserted pre-quarterfinal date or a historical tournament forecast**. For a real forecast, supply the actual pre-quarterfinal cutoff, eight teams, and bracket.

The exporter:

1. Loads the project's trusted local notebook model bundle.
2. Verifies feature-file fingerprints against the bundle's metadata.
3. Rejects a model trained at or after the requested cutoff. A historical cutoff before the model was fitted requires training an earlier model first.
4. Replays only games strictly before the cutoff, freezing state at equal timestamps.
5. Checks replayed pre-match global ratings, local ratings, homes, offsets, and counts against saved notebook features.
6. Exports portable JSON coefficients and rating state. The app does not load browser-uploaded joblib/pickle files.

The app displays the cutoff, model training endpoint, and last source-game timestamp. A cutoff later than available data does not create missing games; inspect that last-data timestamp before treating the package as up to date.

## Set the bracket

Enter a scenario label and eight distinct team identities present in the snapshot. Use a label that identifies hypothetical scenarios clearly. No real participants are filled in automatically.

The ordered quarterfinal pairs define the paths:

```text
QF1 winner ─┐
            ├─ SF1 winner ─┐
QF2 winner ─┘              │
                           ├─ Champion
QF3 winner ─┐              │
            ├─ SF2 winner ─┘
QF4 winner ─┘
```

All seven series are best-of-five. There is no reseeding, draw process, group stage, or Swiss stage. Arrange the quarterfinal pairs to match the supplied semifinal paths; these are declared simulator assumptions, not inferred official rules.

## What each simulation does

For each possible matchup, calculate a game probability from the current state. Use the saved global/international logistic blend when regional features are available; otherwise use global logistic and report the fallback.

One simulated tournament plays the remaining games in each series until one side has three wins, advances that winner, and continues through the final. Repeat this with a seeded random generator. A team's title probability is its championship count divided by the simulation count.

The model and ratings are **frozen inside simulations**. Imagined game outcomes never update global Elo, local ratings, pool offsets, or fitted coefficients. Game outcomes are independent given a fixed matchup probability. Actual entered results can change the state used by the next simulation run.

The output includes semifinal, final, and title probabilities, plus Monte Carlo standard error in percentage points. More simulations reduce sampling noise; they do not measure or eliminate model uncertainty. Results are conditional on the model, supplied bracket, entered results, and assumptions.

## Enter actual game results

Choose an available series, its next game winner, and a timestamp. The timestamp control uses your local time and sends an offset-aware UTC timestamp to the backend. Enter results chronologically, at or after the snapshot cutoff.

- A 2-1 series starts future simulations at 2-1, rather than replaying the completed games.
- At three wins, advancement is fixed and further games in that series are rejected.
- A semifinal/final cannot be played until its qualifying series finish. Its timestamp must be later than both qualifying completion timestamps.
- Actual games update global Elo with K=24 and eligible cross-pool offsets with K=12. Same-pool games and unavailable local strengths do not update offsets.
- Local Elo and domestic homes stay fixed because entered games are international. Logistic coefficients also remain fixed.
- Equal-timestamp actual games read frozen ratings and offsets, then apply summed changes.

The simulator recalculates after a result. Pool-offset updates can affect other teams from the same pool; this follows the current regional calibration model.

## Save, resume, and correct results

For a trained-model run, the default session is saved atomically to `data/processed/simulator_sessions/current.json`. Restarting with the same artifact resumes it. Use `--session PATH` for a separate scenario. The demo is in memory unless you supply a session path.

The interface also downloads a session JSON and imports it for resume. A session contains the bracket, actual-game events, and the exact artifact fingerprint. Resume requires the same model/state artifact. Using another cutoff or model package is rejected rather than silently mixing states.

Every actual game has a unique event ID, series ID, consecutive game number, winner, and timestamp. Duplicate event IDs are rejected. All scores and ratings are rebuilt from the immutable snapshot plus the validated event log, so repeated simulation does not apply actual results twice.

**Undo last game** removes the most recently entered game and replays the remaining events. To correct an earlier result, undo subsequent entries first, or edit and import a downloaded session. Imported events must still satisfy chronology, participant, and advancement rules. Replacing a bracket with existing results requires the interface's explicit clear-results checkbox.

This is bounded session replay for one knockout bracket, not a general historical-data correction or ingestion checkpoint system.

## Code and tests

| File | Responsibility |
| --- | --- |
| `src/prediction.py` | Portable logistic inference, rating-state validation, timestamp-batched Elo/offset updates. |
| `src/export_artifact.py` | Cutoff-safe export and notebook replay verification. |
| `src/tournament.py` | Bracket validation, actual-game replay, exact best-of-five probabilities, Monte Carlo. |
| `src/simulator_app.py` | Loopback HTTP API and atomic session persistence. |
| `web/simulator.html` | Local interactive bracket, game-entry, and probability interface. |
| `tests/` | Synthetic mechanics and HTTP/session integration tests. |

Run the tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover series probabilities against binomial calculations, partial-score conditioning, symmetric fair-team brackets, fixed completed rounds, eliminated teams, seeded reproducibility, nonmutation during simulations, missing-home fallback, timestamp batching, cutoff rejection, invalid/duplicate results, undo, and save/resume. All synthetic fixtures are labeled; they are not match-data or model-evaluation results.

The current real-data export was also checked against every saved final global/local rating, latest home, pool offset, and associated domestic/calibration count. A second earlier cutoff was exported to check that state is reconstructed from a prefix rather than final ratings.

## Current scope

This is a local research prototype bound to `127.0.0.1`, not a deployed public service. Public hosting, multi-user storage, and operational deployment remain separate work. Automated HTTP tests verify the API and persistence; visual browser verification was not completed because computer-use approval timed out.

The model still has exact-name, roster, stale-home, competition-role, missing-data, and development-test limitations described in the main README. Neither simulation count nor a polished interface resolves those modeling limitations. Use a supplied bracket and cutoff before interpreting any real teams' title chances.
