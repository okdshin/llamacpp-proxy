import argparse
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import httpx
import jinja2
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()


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


def load_chat_template(chat_template_path: Optional[str] = None) -> str:
    if chat_template_path:
        return Path(chat_template_path).read_text()
    return settings.chat_template


async def stream_response(response):
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            yield f"{line}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Render template
    template = jinja2.Template(settings.chat_template)
    prompt = template.render(messages=request.messages)

    # Prepare request for llama.cpp server
    llama_request = {
        "prompt": prompt,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "stop": request.stop,
        "stream": request.stream,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{settings.llama_server_url}/v1/completions",
                json=llama_request,
                timeout=300.0,
            )
            response.raise_for_status()

            if request.stream:
                return StreamingResponse(
                    stream_response(response), media_type="text/event-stream"
                )

            # Format non-streaming response
            llama_response = response.json()
            if not isinstance(llama_response, list):
                llama_response = [llama_response]
            print(llama_response)
            completion_response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    CompletionChoice(
                        index=i,
                        message=Message(role="assistant", content=choice["content"]),
                    )
                    for i, choice in enumerate(llama_response)
                ],
                usage={},
            )
            return completion_response

        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Error communicating with llama.cpp server: {str(e)}",
            )


def main():
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

    args = parser.parse_args()

    # Update global settings
    settings.llama_server_url = args.llama_server
    settings.chat_template = load_chat_template(args.chat_template_jinja)

    print(f"Starting server on {args.host}:{args.port}")
    print(f"Proxying requests to {settings.llama_server_url}")
    print(f"Using chat_template from: {args.chat_template_jinja}")
    print(f"```\n{settings.chat_template}\n```")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
