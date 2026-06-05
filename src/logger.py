import json
import logging

class JsonFormatter(logging.Formatter):
    """
    Custom logging Formatter that outputs structured JSON lines.
    """
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt or "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        
        # Capture custom dictionary attributes passed in extra={}
        if hasattr(record, "extra_attrs"):
            log_data.update(record.extra_attrs)
            
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

def get_structured_logger(name="strategyx_api"):
    """
    Configures and returns a logger formatted to output JSON stdout lines.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if logger was already initialized
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False # Prevent doubling up logs in container environments
        
    return logger

api_logger = get_structured_logger()
