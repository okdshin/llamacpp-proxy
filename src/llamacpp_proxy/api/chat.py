import logging
import time
import uuid
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse

from llamacpp_proxy.models.chat import ChatCompletionRequest, ChatCompletionResponse, CompletionChoice, Message
from llamacpp_proxy.services.llama import LlamaClient
from llamacpp_proxy.services.template import TemplateService
from llamacpp_proxy.middleware.auth import get_api_key

logger = logging.getLogger(__name__)

async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Depends(get_api_key),
    llama_client: LlamaClient = Depends(),
    template_service: TemplateService = Depends(),
) -> ChatCompletionResponse:
    """チャット補完APIエンドポイント"""
    logger.info(f"Received request for model: {request.model}")

    try:
        # テンプレートのレンダリング
        prompt = template_service.render(request.messages)

        # llama.cppサーバーへのリクエスト
        llama_request = {
            "prompt": prompt,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n_predict": request.max_tokens,
            "stop": request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
            "stream": request.stream,
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

        return ChatCompletionResponse(
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

    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))