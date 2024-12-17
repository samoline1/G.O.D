from core.models.utility_models import TaskStatus
from fiber.logging_utils import get_logger
from logging import LogRecord, Formatter
from logging.handlers import RotatingFileHandler
from typing import Any, Dict
import json


def create_extra_log(task_id: str | None = None, node_hotkey: str | None = None, status: str | None = None) -> Dict[str, Any]:
    return {"tags": {"task_id": task_id, "status": status, "node_hotkey": node_hotkey}}


class JSONFormatter(Formatter):
    def format(self, record: LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "tags": getattr(record, "tags", {}),
        }
        return json.dumps(log_data)


logger = get_logger(__name__)
file_handler = RotatingFileHandler(
    filename="logs/validator.log",
    maxBytes=100 * 1024 * 1024,  # 100MB
    backupCount=3,
)
file_handler.setFormatter(JSONFormatter())
logger.addHandler(file_handler)
