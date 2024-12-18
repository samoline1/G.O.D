from fiber.logging_utils import get_logger
from logging import LogRecord, Formatter
from logging.handlers import RotatingFileHandler
from typing import Any, Dict
from pathlib import Path
import json
import os


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


def setup_json_logger(name: str):
    print(f"Current working directory: {os.getcwd()}")

    base_dir = Path(__file__).parent.parent.parent
    log_dir = base_dir / "validator" / "logs"

    print(f"Attempting to create directory: {log_dir}")

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Directory created/exists at: {log_dir}")

        log_file = log_dir / "validator.log"
        print(f"Log file will be: {log_file}")

        logger = get_logger(name)
        file_handler = RotatingFileHandler(filename=str(log_file), maxBytes=100 * 1024 * 1024, backupCount=3)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

        logger.info("Logging initialized", extra={"tags": {"setup": "complete"}})
        print(f"Test log written to: {log_file}")

        return logger
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        raise


logger = setup_json_logger(__name__)
