from datetime import datetime, timedelta
from collections import defaultdict
import os
from dataclasses import dataclass
from fastapi import HTTPException
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

@dataclass
class RateLimitSettings:
    window: int = 60  # 60秒
    max_requests: int = 10  # 60秒あたり10リクエスト
    unlimited_api_key: str = os.getenv("UNLIMITED_API_KEY", "")  # 無制限APIキー
    limited_api_key: str = os.getenv("LIMITED_API_KEY", "")  # レート制限付きAPIキー

    def validate(self):
        """設定の検証を行う"""
        if not self.unlimited_api_key and not self.limited_api_key:
            raise ValueError("At least one API key must be configured")


# レート制限のための辞書
rate_limit_store = defaultdict(list)

rate_limit_settings = RateLimitSettings()


def check_rate_limit(api_key: str) -> None:
    """レート制限をチェックする"""
    if api_key == rate_limit_settings.unlimited_api_key:
        return  # 無制限APIキー

    if api_key != rate_limit_settings.limited_api_key:
        return  # 不正なAPIキーは認証時に弾かれるのでここではチェックしない

    # レート制限のチェック
    current_time = datetime.now()
    request_times = rate_limit_store[api_key]

    # 古いリクエスト履歴の削除
    request_times = [
        t
        for t in request_times
        if current_time - t < timedelta(seconds=rate_limit_settings.window)
    ]
    rate_limit_store[api_key] = request_times

    if len(request_times) >= rate_limit_settings.max_requests:
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {rate_limit_settings.max_requests} requests per {rate_limit_settings.window} seconds.",
        )

    # 新しいリクエスト時間の追加
    request_times.append(current_time)