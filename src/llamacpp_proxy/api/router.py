from fastapi import APIRouter, Depends
from llamacpp_proxy.api.chat import chat_completions
from llamacpp_proxy.api.completion import completions
from llamacpp_proxy.middleware.auth import get_api_key

router = APIRouter(prefix="/v1")

router.add_api_route(
    "/chat/completions",
    chat_completions,
    methods=["POST"],
    dependencies=[Depends(get_api_key)]
)

router.add_api_route(
    "/completions",
    completions,
    methods=["POST"],
    dependencies=[Depends(get_api_key)]
)