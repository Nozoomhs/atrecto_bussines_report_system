from clients.backend import LLMBackend
from ollama import chat, ChatResponse
from typing import Optional
from dataclasses import dataclass

@dataclass
class OllamaChatClient(LLMBackend):
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    def complete(self, prompt: str, *, format:str = None) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": self.temperature,
                **({"num_predict": self.max_tokens} if self.max_tokens else {})
            },
        }
        if format is not None:  # ← only add when you really want JSON
            kwargs["format"] = format

        resp: ChatResponse = chat(**kwargs)
        return (resp.message.content or "").strip()