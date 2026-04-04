import json


def load_run(run_file, strict=False):
    meta = None

    episodes = {}
    summaries = {}

    with open(run_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                if strict:
                    raise
                print(f"⚠️ Skipping malformed line {line_num}")
                continue

            entry_type = entry.get("type")

            # ---------- META ----------
            if entry_type == "meta":
                meta = entry
                continue

            # ---------- VALIDATE ----------
            if "episode" not in entry:
                if strict:
                    raise ValueError(f"Missing episode at line {line_num}")
                print(f"⚠️ Skipping entry without episode at line {line_num}")
                continue

            ep = entry["episode"]

            # ---------- STEP ----------
            if entry_type == "step" or entry_type is None:
                episodes.setdefault(ep, []).append(entry)

            # ---------- SUMMARY ----------
            elif entry_type == "episode_summary":
                summaries[ep] = entry

            else:
                if strict:
                    raise ValueError(f"Unknown type '{entry_type}' at line {line_num}")
                print(f"⚠️ Unknown entry type '{entry_type}' at line {line_num}")

    # ---------- SORT STEPS ----------
    for ep in episodes:
        episodes[ep].sort(key=lambda x: x.get("step", 0))

    return {
        "meta": meta,
        "episodes": episodes,
        "summaries": summaries
    }