class LLMBackend:
    """Minimal protocol for chat/complete backends."""

    def complete(self, prompt: str) -> str:
        raise NotImplementedError