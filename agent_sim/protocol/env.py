"""
env.py

Full RL protocol enforcement layer.
"""

import os
import json
import hashlib
from typing import Dict, Any

# -------------------- CONFIG --------------------

PROTOCOL_VERSION = "1.1"

STRICT_PROTOCOL = os.environ.get("STRICT_PROTOCOL", "true").lower() == "true"

GRID_WIDTH = os.environ.get("GRID_WIDTH")
GRID_HEIGHT = os.environ.get("GRID_HEIGHT")

GRID_WIDTH = int(GRID_WIDTH) if GRID_WIDTH else None
GRID_HEIGHT = int(GRID_HEIGHT) if GRID_HEIGHT else None

VALID_ACTIONS = ["up", "down", "left", "right"]

MIN_REWARD = float(os.environ.get("MIN_REWARD", "-inf"))
MAX_REWARD = float(os.environ.get("MAX_REWARD", "inf"))

# -------------------- STATE --------------------

def normalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        raise TypeError("State must be a dictionary")

    required_keys = ["x", "y", "goal"]
    for key in required_keys:
        if key not in state:
            raise ValueError(f"Missing required state key: '{key}'")

    if STRICT_PROTOCOL:
        if not isinstance(state["x"], int):
            raise TypeError("State 'x' must be int (strict mode)")
        if not isinstance(state["y"], int):
            raise TypeError("State 'y' must be int (strict mode)")
        if not isinstance(state["goal"], (list, tuple)):
            raise TypeError("State 'goal' must be list/tuple")
    else:
        state["x"] = int(state["x"])
        state["y"] = int(state["y"])

    goal = list(state["goal"])

    if len(goal) != 2:
        raise ValueError("State 'goal' must have length 2")

    goal = [int(goal[0]), int(goal[1])]

    return {
        "x": state["x"],
        "y": state["y"],
        "goal": goal  # always JSON-safe list
    }


def validate_bounds(state: Dict[str, Any]) -> None:
    x, y = state["x"], state["y"]
    gx, gy = state["goal"]

    if GRID_WIDTH is not None:
        if not (0 <= x < GRID_WIDTH):
            raise ValueError(f"x out of bounds: {x}")
        if not (0 <= gx < GRID_WIDTH):
            raise ValueError(f"goal x out of bounds: {gx}")

    if GRID_HEIGHT is not None:
        if not (0 <= y < GRID_HEIGHT):
            raise ValueError(f"y out of bounds: {y}")
        if not (0 <= gy < GRID_HEIGHT):
            raise ValueError(f"goal y out of bounds: {gy}")


def state_hash(state: Dict[str, Any]) -> str:
    """
    Deterministic hash (JSON-stable)
    """
    normalized = normalize_state(state)

    return hashlib.md5(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# -------------------- ACTION --------------------

def validate_action(action: Any) -> str:
    if not isinstance(action, str):
        raise TypeError("Action must be a string")

    if action not in VALID_ACTIONS:
        raise ValueError(f"Invalid action: {action}")

    return action


# -------------------- REWARD --------------------

def validate_reward(reward: Any) -> float:
    if not isinstance(reward, (int, float)):
        raise TypeError("Reward must be numeric")

    reward = float(reward)

    if reward < MIN_REWARD or reward > MAX_REWARD:
        raise ValueError(f"Reward out of bounds: {reward}")

    return reward


# -------------------- DONE --------------------

def validate_done(done: Any) -> bool:
    if not isinstance(done, bool):
        raise TypeError("Done flag must be boolean")

    return done


# -------------------- STEP CONTRACT --------------------

def to_step(state: Dict[str, Any], reward: Any, done: Any):
    obs = to_observation(state)
    reward = validate_reward(reward)
    done = validate_done(done)

    return obs, reward, done


# -------------------- PUBLIC API --------------------

def to_observation(state: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_state(state)
    validate_bounds(normalized)

    return {
        "x": normalized["x"],
        "y": normalized["y"],
        "goal": normalized["goal"],
        "_protocol_version": PROTOCOL_VERSION
    }