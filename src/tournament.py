"""Fixed eight-team knockout bracket, actual-result replay, and Monte Carlo."""

from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache

import numpy as np

from .prediction import Predictor, RatingState, finite, utc, validate_artifact

SERIES = ("QF1", "QF2", "QF3", "QF4", "SF1", "SF2", "F")
PARENTS = {"SF1": ("QF1", "QF2"), "SF2": ("QF3", "QF4"), "F": ("SF1", "SF2")}


def fingerprint(artifact):
    return hashlib.sha256(json.dumps(artifact, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_bracket(bracket, available_teams):
    if not isinstance(bracket, dict) or not isinstance(bracket.get("label"), str) or not bracket["label"].strip():
        raise ValueError("Give the bracket a label; identify hypothetical scenarios clearly.")
    pairs = bracket.get("quarterfinals")
    if not isinstance(pairs, list) or len(pairs) != 4 or any(not isinstance(p, list) or len(p) != 2 for p in pairs):
        raise ValueError("Supply four quarterfinal pairs.")
    teams = [team for pair in pairs for team in pair]
    if any(not isinstance(t, str) for t in teams) or len(set(teams)) != 8 or not set(teams) <= set(available_teams):
        raise ValueError("Choose eight distinct teams present in the snapshot.")
    return teams


def new_session(artifact, bracket):
    validate_artifact(artifact)
    validate_bracket(bracket, artifact["state"]["global_ratings"])
    return {"schema_version": 1, "artifact_sha256": fingerprint(artifact), "bracket": copy.deepcopy(bracket), "events": []}


def exact_series_probability(probability_a, wins_a=0, wins_b=0):
    p = finite(probability_a, "Probability")
    if not 0 <= p <= 1 or any(type(v) is not int or not 0 <= v <= 3 for v in (wins_a, wins_b)) or wins_a == wins_b == 3:
        raise ValueError("Invalid probability or best-of-five score.")

    @lru_cache(None)
    def recurse(a, b):
        if a == 3:
            return 1.0
        if b == 3:
            return 0.0
        return p * recurse(a + 1, b) + (1 - p) * recurse(a, b + 1)

    return recurse(wins_a, wins_b)


def replay_session(artifact, session):
    """Rebuild from the immutable forecast snapshot; no hidden cumulative state."""
    validate_artifact(artifact)
    if session.get("schema_version") != 1 or session.get("artifact_sha256") != fingerprint(artifact):
        raise ValueError("Session does not belong to this model/state snapshot.")
    teams = validate_bracket(session["bracket"], artifact["state"]["global_ratings"])
    events = session.get("events")
    if not isinstance(events, list) or len(events) > 35:
        raise ValueError("A seven-series best-of-five bracket has at most 35 actual games.")
    state = RatingState.from_dict(artifact["state"])
    scores = {series: [0, 0] for series in SERIES}
    winners, completions, seen = {}, {}, set()
    prior_time = utc(artifact["cutoff_exclusive"])
    batch_time, batch = None, []

    def participants(series):
        if series.startswith("QF"):
            return session["bracket"]["quarterfinals"][int(series[-1]) - 1]
        parents = PARENTS[series]
        return [winners[p] for p in parents] if all(p in winners for p in parents) else None

    for event in events:
        if not isinstance(event, dict) or set(event) != {"event_id", "series", "game_number", "winner", "timestamp"}:
            raise ValueError("Invalid actual-game event fields.")
        event_id, series = event["event_id"], event["series"]
        if not isinstance(event_id, str) or not event_id.strip() or event_id in seen:
            raise ValueError("Actual games require unique event IDs; duplicates are rejected.")
        seen.add(event_id)
        timestamp = utc(event["timestamp"])
        if timestamp < prior_time:
            raise ValueError("Enter actual games in chronological order, at or after the cutoff.")
        if series not in SERIES:
            raise ValueError("Unknown series.")
        if batch_time is not None and timestamp != batch_time:
            state.apply_batch(batch)
            batch = []
        batch_time, prior_time = timestamp, timestamp
        pair = participants(series)
        if pair is None:
            raise ValueError("Both preceding series must finish before this round can be played.")
        if series in PARENTS and any(timestamp <= completions[p] for p in PARENTS[series]):
            raise ValueError("A later round must occur after its qualifying series.")
        score = scores[series]
        if 3 in score:
            raise ValueError("This series is already completed.")
        if type(event["game_number"]) is not int or event["game_number"] != sum(score) + 1:
            raise ValueError("Game numbers must be consecutive within a series.")
        if event["winner"] not in pair:
            raise ValueError("Winner must be a participant in the selected series.")
        side = pair.index(event["winner"])
        score[side] += 1
        batch.append({"team_a": pair[0], "team_b": pair[1], "result_a": int(side == 0), "use_region_calibration": True})
        if score[side] == 3:
            winners[series] = pair[side]
            completions[series] = timestamp
    if batch:
        state.apply_batch(batch)
    predictor = Predictor(**artifact["predictor"])
    details = []
    for series in SERIES:
        pair, score = participants(series), scores[series]
        detail = {"series": series, "teams": pair, "score": score, "winner": winners.get(series)}
        if pair:
            prediction = predictor.predict(state, *pair)
            detail.update(prediction)
            detail["series_probability_a"] = exact_series_probability(prediction["probability_a"], *score)
        details.append(detail)
    return state, details, teams


def add_result(artifact, session, *, event_id, series, winner, timestamp):
    _, details, _ = replay_session(artifact, session)
    found = next((item for item in details if item["series"] == series), None)
    if found is None:
        raise ValueError("Unknown series.")
    candidate = copy.deepcopy(session)
    candidate["events"].append({"event_id": event_id, "series": series, "winner": winner,
                                 "timestamp": timestamp, "game_number": sum(found["score"]) + 1})
    replay_session(artifact, candidate)
    return candidate


def simulate(artifact, session, simulations=100_000, seed=2026):
    if type(simulations) is not int or not 100 <= simulations <= 1_000_000 or type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Use 100 to 1,000,000 simulations and a nonnegative 32-bit seed.")
    state, details, teams = replay_session(artifact, session)
    predictor = Predictor(**artifact["predictor"])
    probability = np.full((8, 8), .5)
    fallback_pairs = []
    for i in range(8):
        for j in range(i + 1, 8):
            prediction = predictor.predict(state, teams[i], teams[j])
            probability[i, j], probability[j, i] = prediction["probability_a"], 1 - prediction["probability_a"]
            if prediction["fallback"]:
                fallback_pairs.append([teams[i], teams[j]])
    rng = np.random.default_rng(seed)
    simulated_winners = {}
    rounds = [np.zeros(8, dtype=int) for _ in range(3)]
    for index, detail in enumerate(details):
        series = detail["series"]
        if index < 4:
            a = np.full(simulations, index * 2, dtype=int)
            b = np.full(simulations, index * 2 + 1, dtype=int)
        else:
            parent_a, parent_b = PARENTS[series]
            a, b = simulated_winners[parent_a], simulated_winners[parent_b]
        wins_a = np.full(simulations, detail["score"][0], dtype=int)
        wins_b = np.full(simulations, detail["score"][1], dtype=int)
        p = probability[a, b]
        for _ in range(5 - sum(detail["score"])):
            active = (wins_a < 3) & (wins_b < 3)
            if not active.any():
                break
            game_a = rng.random(simulations) < p
            wins_a += active & game_a
            wins_b += active & ~game_a
        won = np.where(wins_a == 3, a, b)
        simulated_winners[series] = won
        stage = 0 if index < 4 else 1 if index < 6 else 2
        rounds[stage] += np.bincount(won, minlength=8)
    rows = [{"team": team, "semifinal_probability": rounds[0][i] / simulations,
             "final_probability": rounds[1][i] / simulations, "title_probability": rounds[2][i] / simulations,
             "title_monte_carlo_se": float(np.sqrt((rounds[2][i] / simulations) * (1 - rounds[2][i] / simulations) / simulations))}
            for i, team in enumerate(teams)]
    return {"simulations": simulations, "seed": seed, "teams": sorted(rows, key=lambda row: -row["title_probability"]),
            "series": details, "fallback_pairs": fallback_pairs,
            "assumptions": "Fixed bracket; independent games; constant matchup probabilities; ratings frozen within simulations. Only actual games update ratings."}
