import argparse
import os
import logging
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI

from llamacpp_proxy.config.settings import settings
from llamacpp_proxy.config.rate_limit import rate_limit_settings
from llamacpp_proxy.api.router import router

# Load environment variables
load_dotenv()

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPIアプリケーションの作成
app = FastAPI(
    title="llama.cpp Proxy",
    description="OpenAI API compatible reverse proxy for llama.cpp server",
)

# ルーターの登録
app.include_router(router)


def validate_settings():
    """設定の検証を行う"""
    try:
        settings.validate()
        rate_limit_settings.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="OpenAI API compatible reverse proxy for llama.cpp server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--llamacpp-server",
        default="http://localhost:8080",
        help="URL of the llama.cpp server (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--chat-template-jinja",
        type=str,
        help="Path to chat template file"
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
    settings.llamacpp_server_url = args.llamacpp_server
    settings.chat_template = settings.load_chat_template(args.chat_template_jinja)
    rate_limit_settings.unlimited_api_key = os.getenv("LLAMACPP_PROXY_UNLIMITED_API_KEY")
    rate_limit_settings.limited_api_key = os.getenv("LLAMACPP_PROXY_LIMITED_API_KEY")
    rate_limit_settings.window = args.rate_limit_window
    rate_limit_settings.max_requests = args.rate_limit_max_requests

    # 設定を検証
    validate_settings()

    # 設定情報のログ出力
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Proxying requests to {settings.llamacpp_server_url}")
    logger.info(f"Using chat_template from: {args.chat_template_jinja}")
    logger.info(f"Template content:\n```\n{settings.chat_template}\n```")
    logger.info(
        f"Rate limit configured: {rate_limit_settings.max_requests} requests per {rate_limit_settings.window} seconds"
    )

    if rate_limit_settings.unlimited_api_key:
        logger.info("Unlimited API key is configured")
    if rate_limit_settings.limited_api_key:
        logger.info("Rate-limited API key is configured")
    if not rate_limit_settings.unlimited_api_key and not rate_limit_settings.limited_api_key:
        logger.warning("No API keys are configured")

    # サーバーの起動
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
