from flask import Flask, request, jsonify
from agent_sim.environments.gridworld import GridWorld
from agent_sim.replay.replay import ReplayLogger

app = Flask(__name__)

env = GridWorld()
logger = ReplayLogger()

@app.route("/v1/reset", methods=["POST"])
def reset():
    state = env.reset()

    logger.new_episode()
    logger.log_step(state, action=None, reward=0, done=False)

    return jsonify({
        "state": state
    })

@app.route("/v1/step", methods=["POST"])
def step():
    action = request.json.get("action")

    state, reward, done = env.step(action)

    logger.log_step(state, action, reward, done)

    return jsonify({
        "state": state,
        "reward": reward,
        "done": done
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)