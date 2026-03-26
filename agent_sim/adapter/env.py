def get_state(env):
    """
    Return only the minimal deterministic state
    """
    state = env.state.copy()
    # Optionally filter out unwanted fields:
    filtered_state = {
        "x": state["x"],
        "y": state["y"],
        "goal": state["goal"]
    }
    return filtered_state