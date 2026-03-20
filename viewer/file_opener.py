import subprocess
import platform
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def open_file(path: str) -> bool:
    """
    Open path with OS default app.
    Returns True if file exists and open was attempted, False otherwise.
    """
    if not path or not Path(path).exists():
        logger.warning("File not found: %s", path)
        return False

    system = platform.system()
    try:
        if system == "Linux":
            subprocess.Popen(["xdg-open", path])
        elif system == "Windows":
            os.startfile(path)
        elif system == "Darwin":
            subprocess.Popen(["open", path])
        else:
            logger.error("Unsupported OS: %s", system)
            return False
        return True
    except Exception as e:
        logger.error("Failed to open file %s: %s", path, e)
        return False


def reveal_in_folder(path: str):
    """Open the containing folder in the file manager."""
    if not path:
        return

    parent = str(Path(path).parent)
    system = platform.system()
    try:
        if system == "Linux":
            subprocess.Popen(["xdg-open", parent])
        elif system == "Windows":
            subprocess.Popen(["explorer", f"/select,{path}"])
        elif system == "Darwin":
            subprocess.Popen(["open", "-R", path])
    except Exception as e:
        logger.error("Failed to reveal folder %s: %s", parent, e)


def copy_to_clipboard(root_widget, text: str):
    """Copy text to clipboard using tkinter's clipboard."""
    try:
        root_widget.clipboard_clear()
        root_widget.clipboard_append(text)
    except Exception as e:
        logger.error("Clipboard error: %s", e)
