import logging
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx
from fastapi import HTTPException, Depends

from llamacpp_proxy.config.settings import Settings, settings

logger = logging.getLogger(__name__)

class LlamaClient:
    def __init__(self, settings: Settings = Depends(lambda: settings)):
        self.settings = settings
        self.base_url = settings.llama_server_url

    async def create_completion(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """非ストリーミング補完リクエストを実行"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/completions",
                    json=request,
                    timeout=300.0,
                )
                response.raise_for_status()
                result = response.json()
                return result if isinstance(result, list) else [result]

        except httpx.HTTPError as e:
            logger.error(f"Error communicating with llama.cpp server: {str(e)}")
            raise HTTPException(
                status_code=502,
                detail=f"Error communicating with llama.cpp server: {str(e)}",
            )

    async def create_streaming_completion(
        self, request: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """ストリーミング補完リクエストを実行"""
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/completions",
                    json=request,
                    timeout=300.0,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            yield f"{line}\n\n"

        except httpx.HTTPError as e:
            logger.error(f"Error in streaming completion: {str(e)}")
            raise HTTPException(
                status_code=502,
                detail=f"Error in streaming completion: {str(e)}",
            )