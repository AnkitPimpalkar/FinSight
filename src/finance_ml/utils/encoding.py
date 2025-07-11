import sys
import logging

def ensure_utf8_console():
    """Ensure stdout and logging use UTF-8 encoding to avoid UnicodeEncodeError."""
    # Fix print output
    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    # Fix logging output
    for handler in logging.root.handlers:
        if hasattr(handler, 'stream') and hasattr(handler.stream, 'reconfigure'):
            try:
                handler.stream.reconfigure(encoding='utf-8')
            except Exception:
                pass
