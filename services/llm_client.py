class LLMClient:
    def __init__(self, base_url=None):
        self.base_url = base_url

    def chat(self, prompt, model="agent-default"):
        """
        Stub method (Phase 1).
        Later this will call LiteLLM / Ollama.
        """
        return "right"