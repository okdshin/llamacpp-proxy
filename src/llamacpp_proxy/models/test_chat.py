import pytest
from llamacpp_proxy.models.chat import ChatCompletionRequest, Message

def test_chat_completion_request_default_values():
    request = ChatCompletionRequest(
        model="test-model",
        messages=[Message(role="user", content="test message")]
    )
    assert request.temperature == 0.7
    assert request.top_p == 1.0
    assert request.n == 1
    assert request.stream is False
    assert request.stop is None
    assert request.max_tokens is None
    assert request.presence_penalty == 0.0
    assert request.frequency_penalty == 0.0
    assert request.user is None
    assert request.llamacpp_proxy_grammar is None

def test_chat_completion_request_custom_values():
    request = ChatCompletionRequest(
        model="test-model",
        messages=[Message(role="user", content="test message")],
        temperature=0.5,
        top_p=0.9,
        n=2,
        stream=True,
        stop=["stop1", "stop2"],
        max_tokens=100,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        user="test-user",
        llamacpp_proxy_grammar="test-grammar"
    )
    assert request.temperature == 0.5
    assert request.top_p == 0.9
    assert request.n == 2
    assert request.stream is True
    assert request.stop == ["stop1", "stop2"]
    assert request.max_tokens == 100
    assert request.presence_penalty == 0.1
    assert request.frequency_penalty == 0.2
    assert request.user == "test-user"
    assert request.llamacpp_proxy_grammar == "test-grammar"

def test_message_name_optional():
    message = Message(role="user", content="test message")
    assert message.name is None
    
    message_with_name = Message(role="user", content="test message", name="test-name")
    assert message_with_name.name == "test-name"