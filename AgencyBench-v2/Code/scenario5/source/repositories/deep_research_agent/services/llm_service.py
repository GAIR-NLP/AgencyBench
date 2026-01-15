from typing import List, Dict, AsyncGenerator

from openai import AsyncOpenAI, OpenAI, AzureOpenAI, AsyncAzureOpenAI


class LLMService:
    def __init__(self, model_name, api_key, base_url, provider, api_version):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

        if provider == "openai":
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            self.client_sync = OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif provider == "azure":
            self.client = AsyncAzureOpenAI(api_key=self.api_key, azure_endpoint=self.base_url, api_version=api_version)
            self.client_sync = AzureOpenAI(api_key=self.api_key, azure_endpoint=self.base_url, api_version=api_version)
        else:
            raise ValueError(f"Invalid provider: {provider}")
    
    async def get_chat_completion_stream(
            self, 
            messages: List[Dict[str, str]], 
            max_tokens: int = 1000, 
            temperature: float = 0.7,
            stop: List[str] | str = None,
    ) -> AsyncGenerator[str, None]:
        """Get streaming completion from the LLM"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=True
            )
            
            async for chunk in response:
                if chunk:
                    yield chunk
        except Exception as e:
            print(f"Error in get_chat_completion_stream: {e}")
            raise e
    
    def get_chat_completion_sync(
            self, 
            messages: List[Dict[str, str]], 
            max_tokens: int = 1000, 
            temperature: float = 0.7,
            stop: List[str] | str = None,
    ):
        """Get non-streaming completion from the LLM"""
        response = self.client_sync.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=False
        )

        return response
