from clients.backend import LLMBackend
from dataclasses import dataclass

@dataclass
class OpenAIChatClient(LLMBackend):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 800

    def complete(self, prompt: str) -> str:
        # Lazy import so this file doesn't require the package unless used.
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Install openai >= 1.0: pip install openai") from e

        client = OpenAI()
        # Using a single-message system+user style works well with our PromptParser format
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON matching the schema. No prose."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""