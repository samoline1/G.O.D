import logging
from contextvars import ContextVar
from logging import Logger
from logging import LogRecord
from typing import Optional

from docker.models.containers import Container
from fiber.logging_utils import get_logger as fiber_get_logger


current_context = ContextVar[dict[str, str | dict]]("current_context", default={})


def add_context_tag(key: str, value: str | dict) -> None:
    """Add or update a tag in the current logging context"""
    try:
        context = current_context.get()
        new_context = {**context, key: value}
        current_context.set(new_context)
    except LookupError:
        current_context.set({key: value})


def remove_context_tag(key: str) -> None:
    """Remove a tag from the current logging context"""
    try:
        context = current_context.get()
        if key in context:
            new_context = context.copy()
            del new_context[key]
            current_context.set(new_context)
    except LookupError:
        pass


def clear_context() -> None:
    """
    Removes all tags from the current logging context.
    """
    current_context.set({})


def get_context_tag(key: str) -> Optional[str | dict]:
    """Get a tag value from the current logging context"""
    try:
        context = current_context.get()
        return context.get(key)
    except LookupError:
        return None


def get_all_context_tags() -> dict:
    """Get all tags from the current logging context"""
    try:
        return current_context.get()
    except LookupError:
        return {}


class LogContext:
    def __init__(self, **tags: str | dict):
        self.tags = tags
        self.token = None

    def __enter__(self):
        try:
            current = current_context.get()
            new_context = {**current, **self.tags}
        except LookupError:
            new_context = self.tags
        self.token = current_context.set(new_context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            current_context.reset(self.token)


class ContextTagsFilter(logging.Filter):
    def filter(self, record: LogRecord) -> bool:
        try:
            context = current_context.get()
            for key, value in context.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (bool, str, int, float)):
                            setattr(record, f"ctx_{key}_{sub_key}", str(sub_value))
                elif isinstance(value, (bool, str, int, float)):
                    setattr(record, f"ctx_{key}", str(value))
        except LookupError:
            pass
        return True


def stream_container_logs(container: Container, log_context: dict | None = None):
    logger = get_logger(__name__)

    if not log_context:
        log_context = {}

    with LogContext(**log_context):
        buffer = ""
        try:
            for log_chunk in container.logs(stream=True, follow=True):
                log_text = log_chunk.decode("utf-8", errors="replace")
                buffer += log_text
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line:
                        logger.info(line)
            if buffer:
                logger.info(buffer)
        except Exception as e:
            logger.error(f"Error streaming logs: {str(e)}")


def get_logger(name: str) -> Logger:
    logger = fiber_get_logger(name)
    logger.addFilter(ContextTagsFilter())
    return logger
