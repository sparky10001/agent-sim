# agent-sim
The protocol is the contract. Everything else is replaceable.

```

```
agent-sim
├─ agents
│  ├─ q_agent.py
│  └─ __init__.py
├─ agent_sim
│  ├─ adapter
│  │  ├─ env.py
│  │  ├─ env_interface.py
│  │  ├─ local_env.py
│  │  ├─ remote_env.py
│  │  ├─ server.py
│  │  └─ __init__.py
│  ├─ environments
│  │  ├─ gridworld.py
│  │  └─ __init__.py
│  ├─ replay
│  │  ├─ loader.py
│  │  ├─ renderer.py
│  │  ├─ replay.py
│  │  ├─ summarize.py
│  │  └─ __init__.py
│  ├─ runner
│  │  └─ __init__.py
│  └─ __init__.py
├─ config
├─ docker
│  ├─ vm1
│  ├─ vm2
│  └─ vm3
├─ docker-compose.yml
├─ Dockerfile
├─ Dockerfile.agent
├─ Dockerfile.env
├─ evals
├─ LICENSE
├─ orchestration
│  └─ agent_runner.py
├─ README.md
├─ requirements.txt
└─ services
   └─ llm_client.py

```