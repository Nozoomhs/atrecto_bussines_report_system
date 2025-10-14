from typing import Optional
class LLMBackend:
    """Minimal protocol for chat/complete backends."""

    def complete(self, prompt: str, *, format:Optional[str]) -> str:
        raise NotImplementedError