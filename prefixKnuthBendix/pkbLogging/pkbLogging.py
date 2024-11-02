import logging.handlers
import logging.config
import json
import datetime as dt
import sys


LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class jsonFormatter(logging.Formatter):
    
    def __init__(self, *, format_keys = None):
        logging.Formatter.__init__(self)
        self.format_keys = format_keys if format_keys is not None else {}
    
    def format(self, record):
        message = self._prep_message(record)
        return json.dumps(message, default = str)
    
    def _prep_message(self, record):
        required_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created, tz = dt.timezone.utc).isoformat(),
        }
        if record.exc_info is not None:
            required_fields["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info is not None:
            required_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: msg_val
            if (msg_val := required_fields.pop(val, None)) is not None
            else getattr(record, val)
            for key, val in self.format_keys.items() 
        }
        message.update(required_fields)
        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val
        return message

def make_file_handler(filename, level = None, format_keys = None):
    handler = logging.FileHandler(filename)
    if level is not None:
        handler.setLevel(level)
    formatter = jsonFormatter(format_keys = format_keys)
    handler.setFormatter(formatter)
    return handler

def make_stdout_handler(level = None):
    handler = logging.StreamHandler(stream = sys.stdout)
    if level is not None:
        handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt = "%(levelname)s: %(message)s"))
    return handler