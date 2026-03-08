from openai import AsyncOpenAI

class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    

    async def chat(self, system_prompt: str, user_prompt: str, response_format: dict | None = None) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages= [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format=response_format if response_format else None
        ) 
        return response.choices[0].message.content.strip()