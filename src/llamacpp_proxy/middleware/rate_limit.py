import logging
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from collections import defaultdict

from llamacpp_proxy.config.rate_limit import RateLimitSettings, rate_limit_settings

logger = logging.getLogger(__name__)

# レート制限のための辞書
rate_limit_store = defaultdict(list)

async def check_rate_limit(
    api_key: str,
    settings: RateLimitSettings = Depends(lambda: rate_limit_settings),
) -> None:
    """レート制限をチェックする"""
    if api_key == settings.unlimited_api_key:
        return  # 無制限APIキー

    if api_key != settings.limited_api_key:
        return  # 不正なAPIキーは認証時に弾かれるのでここではチェックしない

    current_time = datetime.now()
    request_times = rate_limit_store[api_key]

    # 古いリクエスト履歴の削除
    request_times = [
        t
        for t in request_times
        if current_time - t < timedelta(seconds=settings.window)
    ]
    rate_limit_store[api_key] = request_times

    if len(request_times) >= settings.max_requests:
        logger.warning(f"Rate limit exceeded for API key: {api_key}")
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {settings.max_requests} requests per {settings.window} seconds.",
        )

    # 新しいリクエスト時間の追加
    request_times.append(current_time)
    logger.debug(f"Request added to rate limit store for API key: {api_key}. Current count: {len(request_times)}")