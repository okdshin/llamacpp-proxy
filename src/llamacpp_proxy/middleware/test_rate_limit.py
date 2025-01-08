import pytest
from datetime import datetime, timedelta
from fastapi import HTTPException
from llamacpp_proxy.middleware.rate_limit import check_rate_limit, rate_limit_store
from llamacpp_proxy.config.rate_limit import RateLimitSettings

@pytest.fixture
def settings():
    return RateLimitSettings(
        window=60,
        max_requests=2,
        unlimited_api_key="test-unlimited",
        limited_api_key="test-limited"
    )

@pytest.fixture(autouse=True)
def clear_rate_limit_store():
    rate_limit_store.clear()
    yield

@pytest.mark.asyncio
async def test_unlimited_api_key(settings):
    # 無制限APIキーは何度でもリクエスト可能
    for _ in range(10):
        await check_rate_limit("test-unlimited", settings)

@pytest.mark.asyncio
async def test_limited_api_key_within_limit(settings):
    # 制限内のリクエストは許可される
    await check_rate_limit("test-limited", settings)
    await check_rate_limit("test-limited", settings)

@pytest.mark.asyncio
async def test_limited_api_key_exceeds_limit(settings):
    # 制限を超えるとエラー
    await check_rate_limit("test-limited", settings)
    await check_rate_limit("test-limited", settings)
    
    with pytest.raises(HTTPException) as exc_info:
        await check_rate_limit("test-limited", settings)
    
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_rate_limit_window_reset(settings):
    # 1回目のリクエスト
    await check_rate_limit("test-limited", settings)
    
    # 古いリクエストを作成（ウィンドウ外）
    old_time = datetime.now() - timedelta(seconds=settings.window + 1)
    rate_limit_store["test-limited"].append(old_time)
    
    # 2回目のリクエスト（古いリクエストは無視される）
    await check_rate_limit("test-limited", settings)

@pytest.mark.asyncio
async def test_invalid_api_key(settings):
    # 不正なAPIキーはレート制限チェックをスキップ（認証で弾かれる）
    await check_rate_limit("invalid-key", settings)