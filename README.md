# agent-sim
The protocol is the contract. Everything else is replaceable.

```
agent-sim
├─ agents
│  ├─ q_agent.py
│  └─ __init__.py
├─ agent_sim
│  ├─ adapter
│  │  ├─ env_interface.py
│  │  ├─ local_env.py
│  │  ├─ remote_env.py
│  │  └─ __init__.py
│  ├─ environments
│  │  ├─ gridworld.py
│  │  └─ __init__.py
│  ├─ protocol
│  │  ├─ env.py
│  │  └─ __init__.py
│  ├─ replay
│  │  ├─ loader.py
│  │  ├─ renderer.py
│  │  ├─ replay.py
│  │  ├─ summarize.py
│  │  └─ __init__.py
│  ├─ server
│  │  ├─ server.py
│  │  └─ __init__.py
│  ├─ validation
│  │  ├─ protocol_validator.py
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
├─ README.md
├─ requirements.txt
├─ runners
│  └─ agent_runner.py
└─ services
   └─ llm_client.py

```