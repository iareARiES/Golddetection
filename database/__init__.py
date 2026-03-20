# database package
# Import db_manager eagerly (no heavy deps), rest lazily
from .db_manager import JewelleryDBManager

__all__ = ["JewelleryDBManager", "GoldImageExtractor", "PostProcessor"]


def __getattr__(name):
    if name == "GoldImageExtractor":
        from .image_extractor import GoldImageExtractor
        return GoldImageExtractor
    if name == "PostProcessor":
        from .post_processor import PostProcessor
        return PostProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
