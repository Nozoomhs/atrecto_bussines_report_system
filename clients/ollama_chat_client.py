from clients.backend import LLMBackend
from ollama import chat, ChatResponse
from typing import Optional
from dataclasses import dataclass

@dataclass
class OllamaChatClient(LLMBackend):
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    def complete(self, prompt: str) -> str:

        resp: ChatResponse = chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": self.temperature,
                **({"num_predict": self.max_tokens} if self.max_tokens else {})
            },
            # This hints Ollama to produce JSON; if the model ignores it we still validate & retry.
            format="json",
        )
        return resp.message.content or ""