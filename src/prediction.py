"""Portable model coefficients and chronological rating state for simulation."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def utc(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Timestamp must be an ISO string including its timezone.")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Use an ISO timestamp including a timezone.") from exc
    if result.tzinfo is None:
        raise ValueError("Timestamp must include a timezone.")
    return result.astimezone(timezone.utc)


def finite(value, name="rating") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def elo_probability(a: float, b: float) -> float:
    return sigmoid((a - b) * math.log(10) / 400)


@dataclass
class RatingState:
    global_ratings: dict[str, float] = field(default_factory=dict)
    local_ratings: dict[str, dict[str, float]] = field(default_factory=dict)
    homes: dict[str, str] = field(default_factory=dict)
    offsets: dict[str, float] = field(default_factory=dict)
    local_games: dict[str, dict[str, int]] = field(default_factory=dict)
    calibration_games: dict[str, int] = field(default_factory=dict)

    def clone(self):
        return copy.deepcopy(self)

    def local(self, team):
        pool = self.homes.get(team)
        return self.local_ratings.get(pool, {}).get(team) if pool else None

    def regional_features(self, a, b):
        local_a, local_b = self.local(a), self.local(b)
        if local_a is None or local_b is None:
            return None
        pool_a, pool_b = self.homes[a], self.homes[b]
        return local_a - local_b, self.offsets.get(pool_a, 0.0) - self.offsets.get(pool_b, 0.0)

    def apply_batch(self, games):
        """Read frozen state for each game, then apply summed changes.

        Each game is a mapping with team_a, team_b, result_a and optional
        rating_pool, updates_home_pool, use_local_elo, use_region_calibration.
        Actual simulator games are international and never update local Elo.
        """
        global_changes, local_changes, pool_changes = {}, {}, {}
        home_changes, domestic_counts, pool_counts = {}, {}, {}
        for game in games:
            a, b, result = game["team_a"], game["team_b"], game["result_a"]
            if not a or not b or a == b or result not in (0, 1):
                raise ValueError("A game needs distinct opponents and a binary result.")
            ga, gb = self.global_ratings.get(a, 1500.0), self.global_ratings.get(b, 1500.0)
            delta = 24 * (result - elo_probability(ga, gb))
            for team, change in [(a, delta), (b, -delta)]:
                global_changes[team] = global_changes.get(team, 0.0) + change

            pool_a, pool_b = self.homes.get(a), self.homes.get(b)
            if game.get("updates_home_pool", False):
                pool_a = pool_b = game.get("rating_pool")
                if not isinstance(pool_a, str) or not pool_a.strip():
                    raise ValueError("Domestic participation requires a pool.")
                for team in (a, b):
                    if team in home_changes and home_changes[team] != pool_a:
                        raise ValueError("Conflicting domestic pools at the same timestamp.")
                    home_changes[team] = pool_a
            initial = 1500.0 if game.get("updates_home_pool", False) else None
            la = self.local_ratings.get(pool_a, {}).get(a, initial) if pool_a else None
            lb = self.local_ratings.get(pool_b, {}).get(b, initial) if pool_b else None
            if game.get("use_local_elo", False):
                if not game.get("updates_home_pool") or not pool_a or pool_a != pool_b:
                    raise ValueError("Local Elo updates require a domestic pool.")
                delta = 24 * (result - elo_probability(la, lb))
                for key, change in [((pool_a, a), delta), ((pool_b, b), -delta)]:
                    local_changes[key] = local_changes.get(key, 0.0) + change
                    domestic_counts[key] = domestic_counts.get(key, 0) + 1
            for pool in (pool_a, pool_b):
                if pool:
                    pool_changes.setdefault(pool, 0.0)
            if game.get("use_region_calibration", False) and pool_a and pool_b and pool_a != pool_b and la is not None and lb is not None:
                strength_a = la + self.offsets.get(pool_a, 0.0)
                strength_b = lb + self.offsets.get(pool_b, 0.0)
                delta = 12 * (result - elo_probability(strength_a, strength_b))
                for pool, change in [(pool_a, delta), (pool_b, -delta)]:
                    pool_changes[pool] += change
                    pool_counts[pool] = pool_counts.get(pool, 0) + 1
        for team, change in global_changes.items():
            self.global_ratings[team] = self.global_ratings.get(team, 1500.0) + change
        for (pool, team), change in local_changes.items():
            ratings = self.local_ratings.setdefault(pool, {})
            ratings[team] = ratings.get(team, 1500.0) + change
            counts = self.local_games.setdefault(pool, {})
            counts[team] = counts.get(team, 0) + domestic_counts[(pool, team)]
        for pool, change in pool_changes.items():
            self.offsets[pool] = self.offsets.get(pool, 0.0) + change
            self.calibration_games[pool] = self.calibration_games.get(pool, 0) + pool_counts.get(pool, 0)
        self.homes.update(home_changes)

    def as_dict(self):
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
            raise ValueError("Invalid rating-state fields.")
        state = cls(**copy.deepcopy(data))
        for name in ("global_ratings", "local_ratings", "homes", "offsets", "local_games", "calibration_games"):
            if not isinstance(getattr(state, name), dict):
                raise ValueError("State fields must be dictionaries.")
        for mapping in (state.global_ratings, state.offsets, *state.local_ratings.values()):
            if not isinstance(mapping, dict):
                raise ValueError("Invalid nested ratings.")
            for key, value in mapping.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("Rating identities must be nonempty strings.")
                finite(value)
        for counts in (state.calibration_games, *state.local_games.values()):
            if not isinstance(counts, dict) or any(type(v) is not int or v < 0 for v in counts.values()):
                raise ValueError("Game counts must be nonnegative integers.")
        for team, pool in state.homes.items():
            if team not in state.global_ratings or not isinstance(pool, str) or not pool:
                raise ValueError("Invalid home membership.")
        return state


@dataclass(frozen=True)
class Predictor:
    global_coefficient: float
    regional_coefficients: tuple[float, float]
    regional_weight: float

    def __post_init__(self):
        finite(self.global_coefficient, "Global coefficient")
        if len(self.regional_coefficients) != 2:
            raise ValueError("Regional model requires two coefficients.")
        for coefficient in self.regional_coefficients:
            finite(coefficient, "Regional coefficient")
        if not 0 <= finite(self.regional_weight, "Blend weight") <= 1:
            raise ValueError("Blend weight must be between zero and one.")

    def predict(self, state, a, b):
        if a == b or a not in state.global_ratings or b not in state.global_ratings:
            raise ValueError("Choose distinct teams present in the snapshot.")
        global_probability = sigmoid(self.global_coefficient * (state.global_ratings[a] - state.global_ratings[b]))
        features = state.regional_features(a, b)
        regional = sigmoid(sum(c * x for c, x in zip(self.regional_coefficients, features))) if features else None
        probability = global_probability if regional is None else (1 - self.regional_weight) * global_probability + self.regional_weight * regional
        return {"probability_a": probability, "global_probability_a": global_probability,
                "regional_probability_a": regional, "fallback": regional is None}

    def as_dict(self):
        return {"global_coefficient": self.global_coefficient, "regional_coefficients": list(self.regional_coefficients),
                "regional_weight": self.regional_weight}


def validate_artifact(data):
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported simulator artifact version.")
    if not {"kind", "cutoff_exclusive", "model_trained_through", "last_history_timestamp", "predictor", "state", "metadata"} <= set(data):
        raise ValueError("Incomplete simulator artifact.")
    cutoff = utc(data["cutoff_exclusive"])
    demo = data.get("kind") == "hypothetical_demo"
    if not demo:
        if data.get("kind") != "trained_model" or utc(data["model_trained_through"]) >= cutoff:
            raise ValueError("Model training must finish before the forecast cutoff.")
    if data.get("last_history_timestamp") and utc(data["last_history_timestamp"]) >= cutoff:
        raise ValueError("Historical state must precede the cutoff.")
    Predictor(**data["predictor"])
    RatingState.from_dict(data["state"])
    return data


def load_artifact(path):
    return validate_artifact(json.loads(Path(path).read_text(encoding="utf-8")))


def demo_artifact():
    """Explicit synthetic example, with no claim to trained-model accuracy."""
    state = RatingState(global_ratings={f"Hypothetical Team {letter}": 1500.0 for letter in "ABCDEFGH"})
    return {"schema_version": 1, "kind": "hypothetical_demo", "cutoff_exclusive": "2000-01-01T00:00:00+00:00",
            "model_trained_through": None, "last_history_timestamp": None,
            "predictor": Predictor(math.log(10) / 400, (math.log(10) / 400,) * 2, .8).as_dict(),
            "state": state.as_dict(), "metadata": {"description": "Synthetic equal-strength teams and illustrative cutoff; Elo probabilities, not a trained model."}}
