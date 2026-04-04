from flask import Flask, request, jsonify
from uuid import uuid4
import time
from threading import Lock

from agent_sim.environments.gridworld import GridWorld
from agent_sim.replay.replay import ReplayLogger
from agent_sim.protocol.env import validate_action, to_step

app = Flask(__name__)

VALID_ACTIONS = ["up", "down", "left", "right"]
MAX_STEPS = 100
SESSION_TTL = 300  # seconds

# -------------------- SESSION STORAGE --------------------

sessions = {}
sessions_lock = Lock()


class Session:
    def __init__(self):
        self.env = GridWorld()
        self.logger = ReplayLogger()
        self.steps = 0
        self.done = False
        self.created_at = time.time()


# -------------------- HEALTH --------------------

@app.route("/v1/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# -------------------- RESET --------------------

@app.route("/v1/reset", methods=["POST"])
def reset():
    session_id = str(uuid4())
    session = Session()

    # Reset environment
    state = session.env.reset()

    # Normalize initial state
    state, _, _ = to_step(state, 0, False)

    # Start replay log
    session.logger.new_episode()
    session.logger.log_step(state, action=None, reward=0, done=False)

    # Store session safely
    with sessions_lock:
        sessions[session_id] = session

    return jsonify({
        "session_id": session_id,
        "state": state
    })


# -------------------- STEP --------------------

@app.route("/v1/step", methods=["POST"])
def step():
    data = request.get_json(silent=True)

    if not data or "action" not in data:
        return jsonify({"error": "Missing 'action'"}), 400

    if "session_id" not in data:
        return jsonify({"error": "Missing 'session_id'"}), 400

    session_id = data["session_id"]

    # Retrieve session safely
    with sessions_lock:
        session = sessions.get(session_id)

    if not session:
        return jsonify({"error": "Invalid session_id"}), 400

    if session.done:
        return jsonify({"error": "Episode already finished"}), 400

    # Validate action
    try:
        action = validate_action(data["action"])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Step environment
    state, reward, done = session.env.step(action)

    # Track steps
    session.steps += 1
    if session.steps >= MAX_STEPS:
        done = True

    session.done = done

    # Normalize output
    try:
        state, reward, done = to_step(state, reward, done)
    except Exception as e:
        return jsonify({
            "error": "protocol_violation",
            "message": str(e)
        }), 500

    # Log replay
    session.logger.log_step(state, action, reward, done)

    return jsonify({
        "session_id": session_id,
        "state": state,
        "reward": reward,
        "done": done
    })


# -------------------- METRICS --------------------

@app.route("/v1/metrics", methods=["GET"])
def metrics():
    with sessions_lock:
        total_sessions = len(sessions)
        total_steps = sum(s.steps for s in sessions.values())

        successes = sum(
            1 for s in sessions.values()
            if s.done
            and hasattr(s.env, "state")
            and hasattr(s.env, "goal")
            and s.env.state["x"] == s.env.goal[0]
            and s.env.state["y"] == s.env.goal[1]
        )

    success_rate = (successes / total_sessions) if total_sessions > 0 else 0

    return jsonify({
        "sessions": total_sessions,
        "steps": total_steps,
        "success_rate": round(success_rate, 3)
    })


# -------------------- CLEANUP --------------------

def cleanup_sessions():
    now = time.time()

    with sessions_lock:
        expired = [
            sid for sid, s in sessions.items()
            if now - s.created_at > SESSION_TTL
        ]

        for sid in expired:
            sessions.pop(sid, None)


@app.before_request
def before_request():
    cleanup_sessions()


# -------------------- ENTRY --------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)