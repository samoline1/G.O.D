from fiber.logging_utils import get_logger

logger = get_logger(__name__)


def stream_logs(container):
    try:
        for log_chunk in container.logs(stream=True, follow=True):
            log_line = log_chunk.decode('utf-8', errors='replace').strip()
            print(log_line)
            logger.info(log_line)
    except Exception as e:
        logger.error(f"Error streaming logs: {str(e)}")

