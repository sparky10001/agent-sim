import os
import sys

def choose_mode():
    print("\n🚀 Select Run Mode:\n")
    print("1) Local Dev (agent_runner)")
    print("2) Remote Single Env")
    print("3) Distributed (multi-env)")
    print("q) Quit\n")

    choice = input("Enter choice: ").strip().lower()

    if choice == "1":
        return "runners.agent_runner", {"ENV_MODE": "local"}
    elif choice == "2":
        base_url = input("Enter BASE_URL (default http://localhost:8000/v1): ").strip()
        base_url = base_url or "http://localhost:8000/v1"
        return "runners.agent_runner", {
            "ENV_MODE": "remote",
            "BASE_URL": base_url
        }
    elif choice == "3":
        urls = input("Enter BASE_URLS (comma-separated): ").strip()
        urls = urls or "http://localhost:8001/v1,http://localhost:8002/v1"
        return "runners.distributed_runner", {
            "BASE_URLS": urls
        }
    elif choice == "q":
        print("👋 Exiting.")
        sys.exit(0)
    else:
        print("❌ Invalid choice\n")
        return choose_mode()


def main():
    # Non-interactive override (Docker / CI)
    runner = os.environ.get("RUNNER")
    if runner:
        print(f"⚙️ Using RUNNER from env: {runner}")
        os.execvp("python", ["python", "-m", runner])
        return

    # Interactive mode
    runner, env_vars = choose_mode()

    # Apply env overrides
    for k, v in env_vars.items():
        os.environ[k] = v

    print(f"\n🚀 Launching {runner}...\n")

    os.execvp("python", ["python", "-m", runner])


if __name__ == "__main__":
    main()