import pytest
from fastapi import HTTPException
from llamacpp_proxy.services.template import TemplateService
from llamacpp_proxy.config.settings import Settings
from llamacpp_proxy.models.chat import Message

@pytest.fixture
def settings():
    return Settings(
        chat_template="""
        {%- for message in messages %}
        {%- if message.role == 'user' %}
        User: {{ message.content }}
        {%- elif message.role == 'assistant' %}
        Assistant: {{ message.content }}
        {%- endif %}
        {%- endfor %}
        """
    )

@pytest.fixture
def template_service(settings):
    return TemplateService(settings)

def test_render_success(template_service):
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
        Message(role="user", content="How are you?"),
    ]
    
    result = template_service.render(messages)
    expected = """
        User: Hello
        Assistant: Hi there
        User: How are you?
        """
    
    # 空白を正規化して比較
    assert result.strip() == expected.strip()

def test_render_template_error(settings):
    # 不正なテンプレート構文
    settings.chat_template = "{{ invalid syntax }"
    template_service = TemplateService(settings)
    
    with pytest.raises(HTTPException) as exc_info:
        template_service.render([Message(role="user", content="test")])
    
    assert exc_info.value.status_code == 400
    assert "Template rendering error" in str(exc_info.value.detail)