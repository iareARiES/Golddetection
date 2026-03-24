import subprocess
import platform
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).parent.parent

def _resolve_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p

def open_file(path: str) -> bool:
    """
    Open path with OS default app.
    Returns True if file exists and open was attempted, False otherwise.
    """
    if not path:
        return False
        
    full_path = _resolve_path(path)
    if not full_path.exists():
        logger.warning("File not found: %s", full_path)
        return False

    system = platform.system()
    path_str = str(full_path)
    try:
        if system == "Linux":
            subprocess.Popen(["xdg-open", path_str])
        elif system == "Windows":
            os.startfile(path_str)
        elif system == "Darwin":
            subprocess.Popen(["open", path_str])
        else:
            logger.error("Unsupported OS: %s", system)
            return False
        return True
    except Exception as e:
        logger.error("Failed to open file %s: %s", path, e)
        return False


def reveal_in_folder(path: str):
    """Open the containing folder in the file manager."""
    full_path = _resolve_path(path)
    if not full_path.exists():
        return
        
    parent_str = str(full_path.parent)
    path_str = str(full_path)
    system = platform.system()
    try:
        if system == "Linux":
            subprocess.Popen(["xdg-open", parent_str])
        elif system == "Windows":
            subprocess.Popen(["explorer", f"/select,{path_str}"])
        elif system == "Darwin":
            subprocess.Popen(["open", "-R", path_str])
    except Exception as e:
        logger.error("Failed to reveal folder %s: %s", parent_str, e)


def copy_to_clipboard(root_widget, text: str):
    """Copy text to clipboard using tkinter's clipboard."""
    try:
        root_widget.clipboard_clear()
        root_widget.clipboard_append(text)
    except Exception as e:
        logger.error("Clipboard error: %s", e)
