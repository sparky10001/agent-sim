import json

def load_run(run_file):
    meta = None
    episodes = {}

    with open(run_file, "r") as f:
        for line in f:
            entry = json.loads(line)

            # Handle metadata
            if entry.get("type") == "meta":
                meta = entry
                continue

            ep = entry["episode"]
            episodes.setdefault(ep, []).append(entry)

    return {
        "meta": meta,
        "episodes": episodes
    }