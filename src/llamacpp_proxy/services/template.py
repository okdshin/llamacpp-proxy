import logging
from typing import List
import jinja2
from fastapi import HTTPException, Depends

from llamacpp_proxy.config.settings import Settings, settings
from llamacpp_proxy.models.chat import Message

logger = logging.getLogger(__name__)

class TemplateService:
    def __init__(self, settings: Settings = Depends(lambda: settings)):
        self.settings = settings
        
    def render(self, messages: List[Message]) -> str:
        """メッセージリストからプロンプトを生成"""
        try:
            template = jinja2.Template(self.settings.chat_template)
            return template.render(messages=messages)
        except jinja2.TemplateError as e:
            logger.error(f"Template rendering error: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Template rendering error: {str(e)}",
            )