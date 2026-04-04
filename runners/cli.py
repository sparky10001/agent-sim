# runners/cli.py
import os
import sys

from runners.single import create_env, train_then_greedy
from runners.distributed import run_distributed
from runners.core import run_experiments

def choose_mode():
    """
    Determines the run mode based on environment variable or user input.
    Returns:
        runner_function: callable
        env_vars: dict of relevant environment variables
    """
    run_mode = os.environ.get("RUN_MODE", "").lower()

    if run_mode in ["local", "remote single", "distributed"]:
        choice = run_mode
    else:
        # Interactive fallback
        print("\n🚀 Select Run Mode:")
        print("1) Local Dev (agent_runner)")
        print("2) Remote Single Env")
        print("3) Distributed (multi-env)")
        print("q) Quit")

        try:
            choice = input("Enter choice: ").strip().lower()
        except EOFError:
            print("❌ No interactive input available. Defaulting to 'local'.")
            choice = "1"

    # Map selection to runner
    if choice in ["1", "local"]:
        runner = run_experiments
        env_vars = {"env_mode": "local"}
    elif choice in ["2", "remote single"]:
        runner = train_then_greedy
        env_vars = {"env_mode": "remote"}
    elif choice in ["3", "distributed"]:
        runner = run_distributed
        env_vars = {"env_mode": "remote"}
    elif choice in ["q", "quit"]:
        print("Exiting.")
        sys.exit(0)
    else:
        print(f"Invalid choice '{choice}', defaulting to local.")
        runner = run_experiments
        env_vars = {"env_mode": "local"}

    return runner, env_vars


def main():
    runner, env_vars = choose_mode()
    os.environ.update({k.upper(): v for k, v in env_vars.items()})

    # Runner may require env object or handle it internally
    if runner in [train_then_greedy, run_experiments]:
        env = create_env()
        if runner == run_experiments:
            runner(env, policy_name="random", num_episodes=10, max_steps=50)
        else:
            runner(env)
    else:
        # Distributed runner handles its own envs
        runner()


if __name__ == "__main__":
    main()