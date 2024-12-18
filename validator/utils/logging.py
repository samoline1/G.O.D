from fiber.logging_utils import get_logger
from logging import LogRecord, Formatter
from logging.handlers import RotatingFileHandler
from typing import Any, Dict
from pathlib import Path
import json


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


def setup_json_logger(name: str):
    log_dir = Path("/root/G.O.D/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(name)
    file_handler = RotatingFileHandler(filename=str(log_dir / "validator.log"), maxBytes=100 * 1024 * 1024, backupCount=3)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    return logger


logger = setup_json_logger(__name__)
