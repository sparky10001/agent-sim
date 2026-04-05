# runners/cli.py
import os
import sys

from runners.single import create_env
from runners.distributed import run_distributed
from runners.core import run_experiments, train_then_greedy

def choose_mode():
    """
    Determines the run mode based on RUN_MODE env or interactive input.
    Returns:
        runner_function: callable
        env_vars: dict of relevant environment variables
    """
    run_mode = os.environ.get("RUN_MODE", "").lower()

    # Map environment variable to canonical choice
    if run_mode in ["local", "remote_single", "distributed"]:
        choice = run_mode
    else:
        # Interactive fallback (for dev only)
        print("\n🚀 Select Run Mode:")
        print("1) Local Dev (agent_runner)")
        print("2) Remote Single Env")
        print("3) Distributed (multi-env)")
        print("q) Quit")
        try:
            choice = input("Enter choice: ").strip().lower()
        except EOFError:
            print("❌ No interactive input available. Defaulting to 'local'.")
            choice = "local"

    # Map selection to runner and environment config
    if choice in ["1", "local"]:
        runner = run_experiments
        env_vars = {"ENV_MODE": "local"}
    elif choice in ["2", "remote_single", "remote single"]:
        runner = train_then_greedy
        env_vars = {"ENV_MODE": "remote"}
    elif choice in ["3", "distributed"]:
        runner = run_distributed
        env_vars = {"ENV_MODE": "remote"}
    elif choice in ["q", "quit"]:
        print("Exiting.")
        sys.exit(0)
    else:
        print(f"Invalid choice '{choice}', defaulting to local.")
        runner = run_experiments
        env_vars = {"ENV_MODE": "local"}

    return runner, env_vars


def main():
    runner, env_vars = choose_mode()
    os.environ.update(env_vars)

    # Run the selected runner
    if runner in [run_experiments, train_then_greedy]:
        env = create_env()
        if runner == run_experiments:
            runner(env, policy_name="random", num_episodes=10, max_steps=50)
        else:
            runner(env)
    else:
        # Distributed runner handles its own environments internally
        runner()


if __name__ == "__main__":
    main()