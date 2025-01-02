import argparse
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import defaultdict

import httpx
import jinja2
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_429_TOO_MANY_REQUESTS, HTTP_401_UNAUTHORIZED

# Load environment variables
load_dotenv()

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# APIキー認証の設定
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


# レート制限の設定
class RateLimitSettings:
    window: int = 60  # 60秒
    max_requests: int = 10  # 60秒あたり10リクエスト
    unlimited_api_key: str = os.getenv("UNLIMITED_API_KEY", "")  # 無制限APIキー
    limited_api_key: str = os.getenv("LIMITED_API_KEY", "")  # レート制限付きAPIキー


rate_limit_settings = RateLimitSettings()

# レート制限のための辞書
rate_limit_store = defaultdict(list)


# Models
class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class CompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]


# Global settings
@dataclass
class Settings:
    llama_server_url: str = ""
    chat_template: str = ""


settings = Settings()


def check_api_key(api_key: str = Depends(api_key_header)):
    """APIキーの検証とレート制限のチェック"""
    if not api_key:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED, detail="API key required"
        )

    # "Bearer "プレフィックスの除去
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]

    # APIキーの検証
    if api_key == rate_limit_settings.unlimited_api_key:
        return api_key  # 無制限APIキー
    elif api_key == rate_limit_settings.limited_api_key:
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
        return api_key
    else:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def validate_settings():
    """設定の検証を行う"""
    if not settings.llama_server_url:
        raise ValueError("llama_server_url must be set")
    if not settings.chat_template:
        raise ValueError("chat_template must be set")
    if (
        not rate_limit_settings.unlimited_api_key
        and not rate_limit_settings.limited_api_key
    ):
        raise ValueError("At least one API key must be configured")


def load_chat_template(chat_template_path: Optional[str] = None) -> str:
    """チャットテンプレートを読み込む"""
    if chat_template_path:
        try:
            return Path(chat_template_path).read_text()
        except Exception as e:
            logger.error(f"Failed to load chat template: {str(e)}")
            raise ValueError(f"Failed to load chat template: {str(e)}")
    return settings.chat_template


async def stream_response(response):
    """ストリーミングレスポンスを処理する"""
    try:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                yield f"{line}\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        raise HTTPException(status_code=500, detail="Streaming error occurred")


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, api_key: str = Depends(check_api_key)
):
    """チャット補完APIエンドポイント"""
    logger.info(f"Received request for model: {request.model}")

    # テンプレートのレンダリング
    try:
        template = jinja2.Template(settings.chat_template)
        prompt = template.render(messages=request.messages)
    except jinja2.TemplateError as e:
        logger.error(f"Template rendering error: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Template rendering error: {str(e)}"
        )

    if not isinstance(request.stop, list):
        request.stop = [request.stop]

    # llama.cppサーバーへのリクエストを準備
    llama_request = {
        "prompt": prompt,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "n_predict": request.max_tokens,
        "stop": request.stop,
        "stream": request.stream,
    }

    logger.info(f"{llama_request=}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{settings.llama_server_url}/completions",
                json=llama_request,
                timeout=300.0,
            )
            response.raise_for_status()

            if request.stream:
                return StreamingResponse(
                    stream_response(response), media_type="text/event-stream"
                )

            # 非ストリーミングレスポンスのフォーマット
            llama_response = response.json()
            if not isinstance(llama_response, list):
                llama_response = [llama_response]

            # レスポンスの内容をログに記録
            logger.debug(f"Llama response: {llama_response}")

            # トークン使用量の計算（実装が必要）
            prompt_tokens = 0  # TODO: implement token counting
            completion_tokens = 0  # TODO: implement token counting
            total_tokens = prompt_tokens + completion_tokens

            completion_response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    CompletionChoice(
                        index=i,
                        message=Message(role="assistant", content=choice["content"]),
                        finish_reason="stop",  # TODO: implement proper finish reason
                    )
                    for i, choice in enumerate(llama_response)
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            )
            return completion_response

        except httpx.HTTPError as e:
            logger.error(f"Error communicating with llama.cpp server: {str(e)}")
            raise HTTPException(
                status_code=502,
                detail=f"Error communicating with llama.cpp server: {str(e)}",
            )


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="OpenAI API compatible reverse proxy for llama.cpp server"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--llama-server",
        default="http://localhost:8080",
        help="URL of the llama.cpp server (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--chat-template-jinja", type=str, help="Path to chat template file"
    )
    parser.add_argument(
        "--rate-limit-window",
        type=int,
        default=60,
        help="Rate limit time window in seconds (default: 60)",
    )
    parser.add_argument(
        "--rate-limit-max-requests",
        type=int,
        default=10,
        help="Maximum number of requests allowed within the time window (default: 10)",
    )

    args = parser.parse_args()

    # グローバル設定を更新
    settings.llama_server_url = args.llama_server
    settings.chat_template = load_chat_template(args.chat_template_jinja)

    # レート制限の設定を更新
    rate_limit_settings.window = args.rate_limit_window
    rate_limit_settings.max_requests = args.rate_limit_max_requests

    # 設定を検証
    try:
        validate_settings()
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        return

    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Proxying requests to {settings.llama_server_url}")
    logger.info(f"Using chat_template from: {args.chat_template_jinja}")
    logger.info(f"Template content:\n```\n{settings.chat_template}\n```")
    logger.info(
        f"Rate limit configured: {rate_limit_settings.max_requests} requests per {rate_limit_settings.window} seconds"
    )

    if rate_limit_settings.unlimited_api_key:
        logger.info("Unlimited API key is configured")
    if rate_limit_settings.limited_api_key:
        logger.info("Rate-limited API key is configured")
    if (
        not rate_limit_settings.unlimited_api_key
        and not rate_limit_settings.limited_api_key
    ):
        logger.warning("No API keys are configured")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
