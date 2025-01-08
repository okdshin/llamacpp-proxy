import logging
from fastapi import HTTPException, Depends
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED
from llamacpp_proxy.config.rate_limit import RateLimitSettings, rate_limit_settings

logger = logging.getLogger(__name__)

# APIキー認証の設定
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(
    api_key: str = Depends(api_key_header),
    rate_limit_settings: RateLimitSettings = Depends(lambda: rate_limit_settings),
) -> str:
    """APIキーを検証する"""
    if not api_key:
        logger.warning("No API key provided")
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )

    # "Bearer "プレフィックスの除去
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]

    # APIキーの検証
    if api_key not in [rate_limit_settings.unlimited_api_key, rate_limit_settings.limited_api_key]:
        logger.warning(f"Invalid API key provided: {api_key}")
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return api_key