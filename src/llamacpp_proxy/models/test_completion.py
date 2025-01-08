import pytest
from llamacpp_proxy.models.completion import CompletionRequest

def test_completion_request_default_values():
    request = CompletionRequest(
        model="test-model",
        prompt="test prompt"
    )
    assert request.suffix is None
    assert request.max_tokens == 16
    assert request.temperature == 0.7
    assert request.top_p == 1.0
    assert request.n == 1
    assert request.stream is False
    assert request.logprobs is None
    assert request.echo is False
    assert request.stop is None
    assert request.presence_penalty == 0.0
    assert request.frequency_penalty == 0.0
    assert request.best_of == 1
    assert request.logit_bias is None
    assert request.user is None
    assert request.llamacpp_proxy_grammar is None

def test_completion_request_custom_values():
    request = CompletionRequest(
        model="test-model",
        prompt=["prompt1", "prompt2"],
        suffix="test suffix",
        max_tokens=100,
        temperature=0.5,
        top_p=0.9,
        n=2,
        stream=True,
        logprobs=5,
        echo=True,
        stop=["stop1", "stop2"],
        presence_penalty=0.1,
        frequency_penalty=0.2,
        best_of=3,
        logit_bias={"token1": 1.0},
        user="test-user",
        llamacpp_proxy_grammar="test-grammar"
    )
    assert request.prompt == ["prompt1", "prompt2"]
    assert request.suffix == "test suffix"
    assert request.max_tokens == 100
    assert request.temperature == 0.5
    assert request.top_p == 0.9
    assert request.n == 2
    assert request.stream is True
    assert request.logprobs == 5
    assert request.echo is True
    assert request.stop == ["stop1", "stop2"]
    assert request.presence_penalty == 0.1
    assert request.frequency_penalty == 0.2
    assert request.best_of == 3
    assert request.logit_bias == {"token1": 1.0}
    assert request.user == "test-user"
    assert request.llamacpp_proxy_grammar == "test-grammar"