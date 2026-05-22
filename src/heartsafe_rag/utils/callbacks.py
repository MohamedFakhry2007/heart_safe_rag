import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class LLMResponseLoggingHandler(BaseCallbackHandler):
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def on_llm_end(self, response: LLMResult, **_: Any) -> None:
        llm_output = response.llm_output or {}
        token_usage = llm_output.get("token_usage", {})
        model_name = llm_output.get("model_name", "unknown")
        texts = []
        for gen_list in response.generations:
            for gen in gen_list:
                texts.append(gen.text[:200])
        self.logger.debug(
            f"LLM response | model={model_name} "
            f"prompt_tokens={token_usage.get('prompt_tokens', '?')} "
            f"completion_tokens={token_usage.get('completion_tokens', '?')} "
            f"texts={texts}"
        )

    def on_llm_error(self, error: Exception, **_: Any) -> None:
        self.logger.error(f"LLM call failed: {error}", exc_info=True)
