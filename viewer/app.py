"""
Gold Detection Viewer — Standalone GUI application.

Run:
    python3 viewer/app.py

Reads from runs/jewellery_detections.db (created by GoldNormal.py).
Can run simultaneously alongside the detection process.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `viewer.*` imports work
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import customtkinter

from viewer.db_reader import DetectionReader
from viewer.file_opener import open_file, reveal_in_folder, copy_to_clipboard
from viewer.widgets.sidebar import Sidebar
from viewer.widgets.topbar import Topbar
from viewer.widgets.table_view import TableView
from viewer.widgets.detail_panel import DetailPanel

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Gold Detection Viewer")
        self.geometry("1200x680")
        self.minsize(900, 560)

        self.db = DetectionReader()
        self.current_filter = "all"
        self.current_search = ""
        self.selected_unique_id = None

        self._build_layout()
        self._refresh()
        self._schedule_auto_refresh()

    def _build_layout(self):
        """Create sidebar, main frame (topbar + table), detail panel side by side."""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar (left) ---
        self.sidebar = Sidebar(self, on_nav_change=self._on_nav_change)
        self.sidebar.grid(row=0, column=0, sticky="nsw")

        # --- Main area (center) ---
        main_frame = customtkinter.CTkFrame(self, fg_color="transparent", corner_radius=0)
        main_frame.grid(row=0, column=1, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        self.topbar = Topbar(main_frame, on_search=self._on_search, on_refresh=self._refresh)
        self.topbar.grid(row=0, column=0, sticky="ew")

        self.table = TableView(main_frame, on_row_select=self._on_row_select)
        self.table.grid(row=1, column=0, sticky="nsew", padx=(0, 0))

        # --- Detail panel (right) ---
        self.detail_panel = DetailPanel(self)
        self.detail_panel.grid(row=0, column=2, sticky="nse")

    def _refresh(self):
        """Re-read DB with current filter + search. Update all widgets."""
        # Get data
        if self.current_search.strip():
            records = self.db.search(self.current_search)
            title_text = f"Search results — {len(records)} record{'s' if len(records) != 1 else ''}"
        else:
            records = self.db.get_all(self.current_filter)
            filter_names = {
                "all": "All detections",
                "pending": "Pending extraction",
                "today": "Today's detections",
                "with_weight": "Detections with weight",
                "duplicates": "Duplicates",
            }
            name = filter_names.get(self.current_filter, "All detections")
            title_text = f"{name} — {len(records)} record{'s' if len(records) != 1 else ''}"

        # Update widgets
        stats = self.db.get_stats()
        self.sidebar.update_stats(stats)
        self.topbar.set_title(title_text)
        self.table.load_data(records)

        # Preserve selected row
        if self.selected_unique_id:
            found = False
            for r in records:
                if r.get("unique_id") == self.selected_unique_id:
                    self.table.selected_id = r.get("id")
                    self.detail_panel.load_record(r)
                    found = True
                    break
            if not found:
                self.selected_unique_id = None
                self.detail_panel.clear()

    def _schedule_auto_refresh(self):
        """Auto-refresh every 5 seconds."""
        self._refresh()
        self.after(5000, self._schedule_auto_refresh)

    def _on_nav_change(self, filter_mode: str):
        self.current_filter = filter_mode
        self.current_search = ""
        self.topbar.search_var.set("")
        self._refresh()

    def _on_search(self, query: str):
        self.current_search = query
        self._refresh()

    def _on_row_select(self, record: dict):
        self.selected_unique_id = record.get("unique_id")
        self.detail_panel.load_record(record)


if __name__ == "__main__":
    app = App()
    app.mainloop()
