import logging
import time
import uuid
from typing import Union, List
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse

from llamacpp_proxy.models.completion import CompletionRequest, CompletionResponse, CompletionResponseChoice
from llamacpp_proxy.services.llama import LlamaClient
from llamacpp_proxy.middleware.auth import get_api_key

logger = logging.getLogger(__name__)

async def completions(
    request: CompletionRequest,
    api_key: str = Depends(get_api_key),
    llama_client: LlamaClient = Depends(),
) -> CompletionResponse:
    """テキスト補完APIエンドポイント"""
    logger.info(f"Received completion request for model: {request.model}")

    try:
        # プロンプトが文字列のリストの場合は最初の要素のみを使用
        prompt = request.prompt[0] if isinstance(request.prompt, list) else request.prompt

        # llama.cppサーバーへのリクエストを準備
        llama_request = {
            "prompt": prompt,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n_predict": request.max_tokens,
            "stop": request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
            "stream": request.stream,
            "n_probs": request.logprobs,  # logprobsのサポート（もしllama.cppサーバーが対応している場合）
        }
        
        if request.llamacpp_proxy_grammar is not None:
            llama_request["grammar"] = request.llamacpp_proxy_grammar

        logger.info(f"{llama_request=}")

        if request.stream:
            # ストリーミングレスポンスの処理
            response = await llama_client.create_streaming_completion(llama_request)
            return StreamingResponse(response, media_type="text/event-stream")

        # 非ストリーミングレスポンスの処理
        llama_response = await llama_client.create_completion(llama_request)
        if not isinstance(llama_response, list):
            llama_response = [llama_response]

        # レスポンスの内容をログに記録
        logger.debug(f"Llama response: {llama_response}")

        # トークン使用量の計算（実装が必要）
        prompt_tokens = 0  # TODO: implement token counting
        completion_tokens = 0  # TODO: implement token counting
        total_tokens = prompt_tokens + completion_tokens

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionResponseChoice(
                    text=choice["content"],
                    index=i,
                    logprobs=None,  # TODO: implement logprobs support
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

    except Exception as e:
        logger.error(f"Error in text completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))