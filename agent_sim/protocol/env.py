"""
env.py

Protocol enforcement layer.

This module defines the canonical environment state format
and provides normalization utilities to ensure ALL environments
(Local, Remote, future implementations) conform to the same contract.

"The protocol is the contract. Everything else is replaceable."
"""

# -------------------- STATE NORMALIZATION --------------------

def normalize_state(state: dict) -> dict:
    """
    Convert raw environment state into canonical protocol format.

    REQUIRED OUTPUT FORMAT:
    {
        "x": int,
        "y": int,
        "goal": (int, int)
    }

    This function is the SINGLE SOURCE OF TRUTH for state shape.
    """

    # 🔒 Enforce required keys
    required_keys = ["x", "y", "goal"]
    for key in required_keys:
        if key not in state:
            raise ValueError(f"Missing required state key: '{key}'")

    # 🔒 Normalize structure
    normalized = {
        "x": int(state["x"]),
        "y": int(state["y"]),
        "goal": tuple(state["goal"])  # ensure consistent type
    }

    return normalized


# -------------------- VALIDATION (OPTIONAL BUT STRONGLY RECOMMENDED) --------------------

def validate_state(state: dict) -> None:
    """
    Validate that a state conforms to the protocol.
    Raises error if invalid.
    """

    if not isinstance(state, dict):
        raise TypeError("State must be a dictionary")

    if not isinstance(state.get("x"), int):
        raise TypeError("State 'x' must be int")

    if not isinstance(state.get("y"), int):
        raise TypeError("State 'y' must be int")

    goal = state.get("goal")
    if not (isinstance(goal, (list, tuple)) and len(goal) == 2):
        raise TypeError("State 'goal' must be a tuple/list of length 2")


# -------------------- PUBLIC API --------------------

def to_observation(state: dict) -> dict:
    """
    Main entry point for all environments.

    Normalizes AND validates state before returning it.
    """
    normalized = normalize_state(state)
    validate_state(normalized)
    return normalized