import pytest
from pathlib import Path
from llamacpp_proxy.config.settings import Settings

def test_validate_empty_settings():
    settings = Settings()
    with pytest.raises(ValueError, match="llama_server_url must be set"):
        settings.validate()

def test_validate_missing_chat_template():
    settings = Settings(llama_server_url="http://localhost:8080")
    with pytest.raises(ValueError, match="chat_template must be set"):
        settings.validate()

def test_validate_valid_settings():
    settings = Settings(
        llama_server_url="http://localhost:8080",
        chat_template="test template"
    )
    settings.validate()  # should not raise

def test_load_chat_template_from_file(tmp_path):
    template_content = "test template content"
    template_file = tmp_path / "test_template.jinja"
    template_file.write_text(template_content)
    
    result = Settings.load_chat_template(str(template_file))
    assert result == template_content

def test_load_chat_template_file_not_found():
    with pytest.raises(ValueError, match="Failed to load chat template"):
        Settings.load_chat_template("nonexistent_file.jinja")