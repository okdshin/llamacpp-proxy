from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class Settings:
    llama_server_url: str = ""
    chat_template: str = ""

    def validate(self):
        """設定の検証を行う"""
        if not self.llama_server_url:
            raise ValueError("llama_server_url must be set")
        if not self.chat_template:
            raise ValueError("chat_template must be set")

    @classmethod
    def load_chat_template(cls, chat_template_path: Optional[str] = None) -> str:
        """チャットテンプレートを読み込む"""
        if chat_template_path:
            try:
                return Path(chat_template_path).read_text()
            except Exception as e:
                logger.error(f"Failed to load chat template: {str(e)}")
                raise ValueError(f"Failed to load chat template: {str(e)}")
        return ""

settings = Settings()