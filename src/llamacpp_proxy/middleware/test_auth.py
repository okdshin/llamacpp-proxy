import pytest
from fastapi import HTTPException
from llamacpp_proxy.middleware.auth import get_api_key
from llamacpp_proxy.config.rate_limit import RateLimitSettings

@pytest.fixture
def rate_limit_settings():
    return RateLimitSettings(
        unlimited_api_key="test-unlimited",
        limited_api_key="test-limited"
    )

@pytest.mark.asyncio
async def test_get_api_key_no_key():
    with pytest.raises(HTTPException) as exc_info:
        await get_api_key(None)
    
    assert exc_info.value.status_code == 401
    assert "API key required" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_get_api_key_invalid(rate_limit_settings):  # フィクスチャを引数として受け取る
    with pytest.raises(HTTPException) as exc_info:
        await get_api_key("invalid-key", rate_limit_settings)
    
    assert exc_info.value.status_code == 401
    assert "Invalid API key" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_get_api_key_unlimited(rate_limit_settings):
    result = await get_api_key("test-unlimited", rate_limit_settings)
    assert result == "test-unlimited"

@pytest.mark.asyncio
async def test_get_api_key_limited(rate_limit_settings):
    result = await get_api_key("test-limited", rate_limit_settings)
    assert result == "test-limited"

@pytest.mark.asyncio
async def test_get_api_key_with_bearer(rate_limit_settings):
    result = await get_api_key("Bearer test-unlimited", rate_limit_settings)
    assert result == "test-unlimited"