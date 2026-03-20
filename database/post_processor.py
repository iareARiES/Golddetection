import threading
import time
import logging

logger = logging.getLogger(__name__)


class PostProcessor:
    """Background post-processor that extracts gold images from saved videos."""

    def __init__(self, db_manager, extractor):
        """
        :param db_manager: JewelleryDBManager instance
        :param extractor: GoldImageExtractor instance
        """
        self.db = db_manager
        self.extractor = extractor
        self._stop_event = threading.Event()

    def process_pending(self):
        """
        Get all records where image_path IS NULL.
        For each, run extractor.extract_best_frame(video_path).
        If successful, call db_manager.update_image_path(row_id, image_path).
        Log success/failure.
        """
        pending = self.db.get_pending_image_extraction()
        if not pending:
            return

        logger.info("Post-processor: %d pending video(s) to process", len(pending))

        for record in pending:
            video_path = record["video_path"]
            row_id = record["id"]

            try:
                image_path = self.extractor.extract_best_frame(video_path)
                if image_path:
                    self.db.update_image_path(row_id, image_path)
                else:
                    logger.warning("No image extracted for row %d: %s", row_id, video_path)
            except Exception as e:
                logger.error("Post-processing failed for row %d: %s", row_id, e)

    def _run_loop(self):
        """Internal loop: process pending every 15 seconds until stopped."""
        while not self._stop_event.is_set():
            try:
                self.process_pending()
            except Exception as e:
                logger.error("Post-processor loop error: %s", e)
            self._stop_event.wait(15)

    def start_background_thread(self):
        """
        Start a daemon thread that calls process_pending() every 15 seconds.
        Uses threading.Thread with daemon=True.
        """
        thread = threading.Thread(target=self._run_loop, daemon=True)
        thread.start()
        logger.info("Post-processor background thread started")

    def stop(self):
        """Signal the background thread to stop."""
        self._stop_event.set()
