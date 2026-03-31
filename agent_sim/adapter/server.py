from flask import Flask, request, jsonify
from agent_sim.environments.gridworld import GridWorld
from agent_sim.replay.replay import ReplayLogger

app = Flask(__name__)

env = GridWorld()
logger = ReplayLogger()

VALID_ACTIONS = ["up", "down", "left", "right"]

# -------------------- HEALTH --------------------

@app.route("/v1/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok"
    }), 200


# -------------------- RESET --------------------

@app.route("/v1/reset", methods=["POST"])
def reset():
    state = env.reset()

    logger.new_episode()
    logger.log_step(state, action=None, reward=0, done=False)

    return jsonify({
        "state": state
    })


# -------------------- STEP --------------------

@app.route("/v1/step", methods=["POST"])
def step():
    data = request.get_json(silent=True)

    if not data or "action" not in data:
        return jsonify({"error": "Missing 'action'"}), 400

    action = data["action"]

    if action not in VALID_ACTIONS:
        return jsonify({"error": "Invalid action"}), 400

    state, reward, done = env.step(action)

    logger.log_step(state, action, reward, done)

    return jsonify({
        "state": state,
        "reward": reward,
        "done": done
    })


# -------------------- METRICS (NEW) --------------------

@app.route("/v1/metrics", methods=["GET"])
def metrics():
    """
    Basic runtime metrics.
    Assumes ReplayLogger stores episodes as a list of steps.
    """

    try:
        episodes = getattr(logger, "episodes", [])

        total_episodes = len(episodes)
        total_steps = sum(len(ep) for ep in episodes)

        successes = 0
        total_reward = 0

        for ep in episodes:
            ep_reward = sum(step.get("reward", 0) for step in ep)
            total_reward += ep_reward

            if any(step.get("reward", 0) > 0 for step in ep):
                successes += 1

        success_rate = (successes / total_episodes) if total_episodes > 0 else 0
        avg_reward = (total_reward / total_episodes) if total_episodes > 0 else 0

        return jsonify({
            "episodes": total_episodes,
            "steps": total_steps,
            "success_rate": round(success_rate, 3),
            "avg_reward": round(avg_reward, 3)
        }), 200

    except Exception as e:
        return jsonify({
            "error": "metrics_failed",
            "message": str(e)
        }), 500


# -------------------- ENTRY --------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)