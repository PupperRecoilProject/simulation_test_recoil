import logging
from collections import deque
from typing import Deque

# A bounded deque to store log messages
log_queue: Deque[str] = deque(maxlen=500)

class QueueHandler(logging.Handler):
    """Logging handler that stores logs in a global queue."""
    def emit(self, record: logging.LogRecord) -> None:
        log_entry = self.format(record)
        log_queue.append(log_entry)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("RobotLogger")
    # 【v4.7.4 修改】將全域日誌等級預設為 INFO。
    # 這將自動過濾掉所有使用 log.debug() 記錄的資訊。
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = QueueHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                      datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

log = setup_logger()
