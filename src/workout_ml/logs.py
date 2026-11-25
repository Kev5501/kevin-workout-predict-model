"""JSON logs that Cloud Logging parses into severity and fields."""
from __future__ import annotations

import json
import logging
import sys

_STANDARD = set(logging.LogRecord('', 0, '', 0, '', (), None).__dict__) | {'message', 'asctime', 'taskName'}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {'severity': record.levelname, 'message': record.getMessage(), 'logger': record.name}
        out |= {k: v for k, v in record.__dict__.items() if k not in _STANDARD}
        if record.exc_info:
            out['exception'] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging(level: str = 'INFO') -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
