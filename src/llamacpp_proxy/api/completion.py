import logging
import time
import uuid
import math
from typing import Any, Union, List, Dict
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse

from llamacpp_proxy.models.completion import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    LogProbs
)
from llamacpp_proxy.services.llamacpp import LlamaCppClient
from llamacpp_proxy.middleware.auth import get_api_key

logger = logging.getLogger(__name__)

def process_logprobs(token_data: List[Dict[str, Any]], top_n: int) -> LogProbs:
    """トークンの確率情報を処理してLogProbsオブジェクトを生成"""
    tokens = []
    token_logprobs = []
    top_logprobs = []
    text_offset = []  # 現在の実装では正確な文字オフセットは計算しない
    
    current_offset = 0
    for token in token_data:
        tokens.append(token["text"])
        token_logprob = math.log(token["probability"]) if token["probability"] > 0 else float("-inf")
        token_logprobs.append(token_logprob)
        
        # top_logprobsの処理
        assert "top_probs" in token
        top_probs = {
            prob["text"]: math.log(prob["probability"]) if prob["probability"] > 0 else float("-inf")
            for prob in token["top_probs"][:top_n]
        }
        top_logprobs.append(top_probs)
        
        # 簡易的な文字オフセットの計算（正確ではありません）
        text_offset.append(current_offset)
        current_offset += len(token["text"])
    
    return LogProbs(
        tokens=tokens,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs,
        text_offset=text_offset
    )

async def completions(
    request: CompletionRequest,
    api_key: str = Depends(get_api_key),
    llamacpp_client: LlamaCppClient = Depends(),
) -> CompletionResponse:
    """テキスト補完APIエンドポイント"""
    logger.info(f"Received completion request for model: {request.model}")

    try:
        # プロンプトが文字列のリストの場合は最初の要素のみを使用
        prompt = request.prompt[0] if isinstance(request.prompt, list) else request.prompt

        # llama.cppサーバーへのリクエストを準備
        llamacpp_request = {
            "prompt": prompt,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n_predict": request.max_tokens,
            "stop": request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
            "stream": request.stream,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }
        
        # logprobsが指定されている場合、n_probsを設定
        if request.logprobs is not None:
            llamacpp_request["n_probs"] = request.logprobs
        
        if request.llamacpp_proxy_grammar is not None:
            llamacpp_request["grammar"] = request.llamacpp_proxy_grammar

        logger.info(f"{llamacpp_request=}")

        if request.stream:
            # ストリーミングレスポンスの処理
            response = await llamacpp_client.create_streaming_completion(llamacpp_request)
            return StreamingResponse(response, media_type="text/event-stream")

        # 非ストリーミングレスポンスの処理
        llamacpp_response = await llamacpp_client.create_completion(llamacpp_request)
        if not isinstance(llamacpp_response, list):
            llamacpp_response = [llamacpp_response]

        # レスポンスの内容をログに記録
        logger.debug(f"Response: {llamacpp_response}")

        # トークン使用量の計算（実装が必要）
        prompt_tokens = 0  # TODO: implement token counting
        completion_tokens = 0  # TODO: implement token counting
        total_tokens = prompt_tokens + completion_tokens

        choices = []
        for i, choice in enumerate(llamacpp_response):
            logprobs = None
            if request.logprobs is not None and "tokens" in choice:
                logprobs = process_logprobs(choice["tokens"], request.logprobs)

            choices.append(
                CompletionResponseChoice(
                    text=choice["content"],
                    index=i,
                    logprobs=logprobs,
                    finish_reason="stop",  # TODO: implement proper finish reason
                )
            )

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )

    except Exception as e:
        logger.error(f"Error in text completion: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "internal_server_error"
                }
            }
        )
