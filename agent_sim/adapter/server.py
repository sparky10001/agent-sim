from flask import Flask, request, jsonify
from agent_sim.environments.gridworld import GridWorld

app = Flask(__name__)
env = GridWorld()

@app.route("/v1/reset", methods=["POST"])
def reset():
    return jsonify(env.reset())

@app.route("/v1/step", methods=["POST"])
def step():
    action = request.json.get("action")
    state, reward, done = env.step(action)

    return jsonify({
        "state": state,
        "reward": reward,
        "done": done
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)