import pytest
from unittest.mock import AsyncMock, patch
import httpx
from fastapi import HTTPException
from llamacpp_proxy.services.llama import LlamaClient
from llamacpp_proxy.config.settings import Settings

@pytest.fixture
def settings():
    return Settings(llama_server_url="http://test-server:8080")

@pytest.fixture
def client(settings):
    return LlamaClient(settings)

@pytest.mark.asyncio
async def test_create_completion_success(client):
    mock_response = {"content": "test response"}
    
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock(
            status_code=200,
            json=lambda: mock_response,
            raise_for_status=lambda: None
        )
        
        result = await client.create_completion({"prompt": "test"})
        assert result == [mock_response]

@pytest.mark.asyncio
async def test_create_completion_http_error(client):
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = httpx.HTTPError("Test error")
        
        with pytest.raises(HTTPException) as exc_info:
            await client.create_completion({"prompt": "test"})
        
        assert exc_info.value.status_code == 502
        assert "Error communicating with llama.cpp server" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_create_streaming_completion_success(client):
    async def mock_aiter():
        for line in ["data: line1", "data: line2"]:
            yield line

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = AsyncMock()
    mock_response.aiter_lines = mock_aiter

    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_stream.return_value.__aenter__.return_value = mock_response
        
        result = []
        async for line in client.create_streaming_completion({"prompt": "test"}):
            result.append(line)
        
        assert result == ["data: line1\n\n", "data: line2\n\n"]

@pytest.mark.asyncio
async def test_create_streaming_completion_http_error(client):
    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_stream.side_effect = httpx.HTTPError("Test error")
        
        with pytest.raises(HTTPException) as exc_info:
            async for _ in client.create_streaming_completion({"prompt": "test"}):
                pass
        
        assert exc_info.value.status_code == 502
        assert "Error in streaming completion" in str(exc_info.value.detail)