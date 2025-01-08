from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    # これらのパラメータは現在サポートしていない
    echo: Optional[bool] = Field(None, description="Not supported")
    suffix: Optional[str] = Field(None, description="Not supported")
    best_of: Optional[int] = Field(None, description="Not supported")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Not supported")
    logprobs: Optional[int] = Field(None, description="Not supported")
    user: Optional[str] = None

    # extra parameter for llamacpp
    llamacpp_proxy_grammar: Optional[str] = None

    @validator('n')
    def validate_n(cls, v):
        if v != 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "Only n=1 is supported",
                        "type": "invalid_request_error",
                        "param": "n",
                        "code": "parameter_not_supported"
                    }
                }
            )
        return v

    @validator('echo', 'suffix', 'best_of', 'logit_bias', 'logprobs')
    def validate_unsupported_params(cls, v, values, field):
        if v is not None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"Parameter {field.name} is not supported",
                        "type": "invalid_request_error",
                        "param": field.name,
                        "code": "parameter_not_supported"
                    }
                }
            )
        return v

class CompletionResponseChoice(BaseModel):
    text: str
    index: int
    finish_reason: Optional[str] = None

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionResponseChoice]
    usage: Dict[str, int]