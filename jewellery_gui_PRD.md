# PRD: Jewellery Detection — GUI Viewer Application

**Project:** Gold Detection System — GUI Viewer  
**Target IDE:** Anthropic Claude Opus (Antigravity IDE)  
**Tech Stack:** Python + CustomTkinter + SQLite3 + Pillow + subprocess  
**Document Type:** Product Requirements Document (Prompt/Spec for Coding Agent)  
**Version:** 1.0  

---

## 🧠 Context for the Coding Agent

This GUI is a **standalone viewer** for the jewellery detection system. It reads from the existing SQLite database at `runs/jewellery_detections.db` (schema defined in the Database PRD). It does NOT do any detection — it only reads, displays, and lets the operator open files.

**It must run independently from the main detection process.** Both can run simultaneously — the viewer auto-refreshes from the live database.

---

## 🎨 Tech Stack Decision

| Concern | Choice | Reason |
|---|---|---|
| UI framework | **CustomTkinter** | Native look, no browser, runs on Raspberry Pi, easy pip install |
| Database | **sqlite3** (stdlib) | Already used in backend, no extra dependency |
| Image display | **Pillow (PIL)** | Resize thumbnails, display JPEGs inline |
| File opening | **subprocess + platform** | `xdg-open` on Linux/RPi, `start` on Windows |
| Auto-refresh | **threading.Timer** | Poll DB every 5 seconds without blocking UI |

**Install requirements:**
```bash
pip install customtkinter pillow
```

---

## 📐 Application Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  SIDEBAR (200px)  │           MAIN TABLE               │  DETAIL    │
│                   │                                     │  PANEL     │
│  [logo / title]   │  [topbar: filter pills + search]   │  (280px)   │
│                   │  ─────────────────────────────────  │            │
│  ● All detections │  thumb | ID | Date | Weight | Stat  │  [image]   │
│  ○ Pending        │  ─────────────────────────────────  │            │
│  ○ Duplicates     │  row  row  row  row  row  row ...   │  [details] │
│                   │                                     │            │
│  ─────────────── │                                     │  [buttons] │
│  Stats: 47 total  │                                     │            │
└─────────────────────────────────────────────────────────────────────┘
```

**Window size:** 1200 × 680px minimum, resizable  
**Title:** `Gold Detection Viewer`

---

## 🗂 File Structure

```
project_root/
├── viewer/
│   ├── __init__.py
│   ├── app.py               ← Main entry point: run this file
│   ├── db_reader.py         ← SQLite read-only queries
│   ├── file_opener.py       ← Cross-platform open logic
│   ├── widgets/
│   │   ├── sidebar.py       ← Left nav panel
│   │   ├── table_view.py    ← Main detection table (scrollable)
│   │   ├── detail_panel.py  ← Right-side record detail
│   │   └── topbar.py        ← Filter pills + search box
```

Run with:
```bash
python viewer/app.py
```

---

## 📦 Module Specifications

---

### `viewer/db_reader.py`

**Class: `DetectionReader`**

Read-only interface to the database. Never writes. Opens a new connection per call (safe for concurrent access while main detection process also writes).

```python
class DetectionReader:
    def __init__(self, db_path: str = "runs/jewellery_detections.db"):
        self.db_path = db_path

    def get_all(self, filter_mode: str = "all") -> list[dict]:
        """
        filter_mode options:
          "all"      → all rows, order by captured_at DESC
          "pending"  → rows where image_path IS NULL
          "today"    → rows where date(captured_at) = date('now')
          "with_weight" → rows where weight_grams != 'unknown'
          "duplicates" → rows where is_duplicate = 1
        Returns list of dicts with keys:
          id, unique_id, video_path, image_path, weight_grams,
          captured_at, image_extracted_at, is_duplicate, notes
        """

    def search(self, query: str) -> list[dict]:
        """
        Full text search across: unique_id, weight_grams, notes.
        Returns same dict structure as get_all().
        Uses SQL LIKE with % wildcards.
        """

    def get_stats(self) -> dict:
        """
        Return: {
          "total": int,
          "pending_image": int,
          "duplicates": int,
          "today_count": int
        }
        """

    def get_by_id(self, row_id: int) -> dict | None:
        """Fetch a single record by primary key id."""
```

---

### `viewer/file_opener.py`

**Function: `open_file(path: str)`**

Opens a file with the system default application. Works on Windows, Linux, and Raspberry Pi (Raspbian).

```python
import subprocess
import platform
import os
from pathlib import Path

def open_file(path: str) -> bool:
    """
    Open path with OS default app.
    - Linux/RPi: subprocess.Popen(['xdg-open', path])
    - Windows: os.startfile(path)
    - macOS: subprocess.Popen(['open', path])
    Returns True if file exists and open was attempted, False otherwise.
    Logs a warning if file does not exist.
    """

def reveal_in_folder(path: str):
    """
    Open the containing folder in the file manager.
    - Linux: xdg-open on the parent directory
    - Windows: explorer /select,<path>
    - macOS: open -R <path>
    """

def copy_to_clipboard(root_widget, text: str):
    """Use root_widget.clipboard_clear() + clipboard_append() to copy text."""
```

---

### `viewer/widgets/sidebar.py`

**Class: `Sidebar(customtkinter.CTkFrame)`**

Left navigation panel. Fixed 200px width.

**UI elements:**
- App title: `"Gold Detection"` (14px bold) + subtitle `"Jewellery System"` (11px muted)
- Navigation items (clickable): `All Detections`, `Pending Extraction`, `Today`, `With Weight`, `Duplicates`
- Active item highlighted with left accent border (2px `#378ADD`) and bold text
- Stats section at bottom (separated by horizontal line):
  - Total: `{n}` detections
  - Pending: `{n}` awaiting image
  - Today: `{n}` today

**Callbacks:**
- `on_nav_change(filter_mode: str)` — called when user clicks a nav item, triggers table refresh

```python
class Sidebar(customtkinter.CTkFrame):
    def __init__(self, parent, on_nav_change: callable, **kwargs):
        ...

    def update_stats(self, stats: dict):
        """Refresh the stat numbers at the bottom."""
```

---

### `viewer/widgets/topbar.py`

**Class: `Topbar(customtkinter.CTkFrame)`**

Horizontal bar above the table. Contains:
- Section title (dynamic, e.g., `"All detections — 47 records"`)
- Search box (real-time, triggers on each keystroke after 300ms debounce)
- Refresh button (`⟳ Refresh`) that manually triggers a DB re-read

```python
class Topbar(customtkinter.CTkFrame):
    def __init__(self, parent, on_search: callable, on_refresh: callable, **kwargs):
        ...

    def set_title(self, title: str):
        """Update the section title label."""
```

**Search debounce:** Use `after(300, callback)` on each keystroke, cancelling the previous `after` call. Don't fire DB queries on every character.

---

### `viewer/widgets/table_view.py`

**Class: `TableView(customtkinter.CTkScrollableFrame)`**

The main scrollable table of detections.

**Columns and widths:**

| Column | Width | Content |
|---|---|---|
| Thumbnail | 60px | Small `CTkImage` of the gold (from `image_path`). If not available, show gray placeholder with `?` |
| Unique ID | 200px | Monospace, truncated to 24 chars, full ID as tooltip |
| Date & Time | 150px | Formatted as `DD Mon YYYY, HH:MM:SS` |
| Weight | 80px | `{n} g` or `unknown` badge |
| Video | 70px | `✓ saved` green badge or `✗ missing` red badge |
| Image | 80px | `✓ ready` green / `⏳ processing` amber / `✗ missing` red |
| Actions | 120px | `▶ Video` button + `⊞ Image` button |

**Row behavior:**
- Clicking anywhere on a row → selects it and populates the detail panel
- Selected row has a light blue background
- Hovering shows a slightly darker background
- Double-clicking a row → opens the image directly (if available), else opens video

**Thumbnail loading:**
- Load images in a background thread to avoid freezing the UI
- Resize to `60×44px` maintaining aspect ratio using Pillow
- Cache loaded thumbnails by `image_path` in a dict to avoid reloading

```python
class TableView(customtkinter.CTkScrollableFrame):
    def __init__(self, parent, on_row_select: callable, **kwargs):
        ...

    def load_data(self, records: list[dict]):
        """Clear and re-render all rows from records list."""

    def clear_selection(self):
        """Deselect all rows."""

    def _load_thumbnail_async(self, image_path: str, label_widget):
        """Load image in thread, update label on main thread via after()."""
```

---

### `viewer/widgets/detail_panel.py`

**Class: `DetailPanel(customtkinter.CTkFrame)`**

Right panel. Shows detail for the selected detection. 280px wide.

**Sections:**

**1. Image preview**
- Large `CTkImage` display, aspect ratio 16:10, fills panel width
- If `image_path` is available: show extracted best frame
- If not: show placeholder with text `"Image processing..."`
- Below image: small label `"best frame extracted"` or `"snapshot"`

**2. Record details** (label-value pairs)

| Label | Value |
|---|---|
| Unique ID | monospace font, truncated |
| Weight | `{n} g` |
| Captured | `HH:MM:SS` |
| Date | `DD Mon YYYY` |
| Duration | time between detection start and last_detection_time (if stored) |
| Status | green `complete` / amber `processing` badge |

**3. Action buttons** (stacked, full width)

- `▶ Open video` (primary, filled blue) → `open_file(record['video_path'])`
- `⊞ Open image` → `open_file(record['image_path'])` (disabled if image_path is None)
- `⎘ Copy unique ID` → copies `unique_id` to clipboard
- `📁 Reveal in folder` → `reveal_in_folder(record['video_path'])`

**Empty state:** When no row is selected, show centered muted text: `"Select a detection to view details"`

```python
class DetailPanel(customtkinter.CTkFrame):
    def __init__(self, parent, file_opener, **kwargs):
        ...

    def load_record(self, record: dict):
        """Populate all fields and image for this record."""

    def clear(self):
        """Reset to empty state."""
```

---

### `viewer/app.py`

**Class: `App(customtkinter.CTk)`**

Main application window. Wires all widgets together.

```python
import customtkinter
from viewer.db_reader import DetectionReader
from viewer.file_opener import open_file, reveal_in_folder, copy_to_clipboard
from viewer.widgets.sidebar import Sidebar
from viewer.widgets.topbar import Topbar
from viewer.widgets.table_view import TableView
from viewer.widgets.detail_panel import DetailPanel

customtkinter.set_appearance_mode("System")   # auto dark/light
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

        self._build_layout()
        self._refresh()
        self._schedule_auto_refresh()

    def _build_layout(self):
        """Create sidebar, main frame (topbar + table), detail panel side by side."""

    def _refresh(self):
        """
        Re-read DB with current filter + search.
        Update sidebar stats, topbar title, table rows.
        """

    def _schedule_auto_refresh(self):
        """Call self._refresh() every 5000ms using self.after()."""

    def _on_nav_change(self, filter_mode: str):
        self.current_filter = filter_mode
        self.current_search = ""
        self._refresh()

    def _on_search(self, query: str):
        self.current_search = query
        self._refresh()

    def _on_row_select(self, record: dict):
        self.detail_panel.load_record(record)

if __name__ == "__main__":
    app = App()
    app.mainloop()
```

---

## ⚙️ Auto-Refresh Behavior

- The app polls the DB every **5 seconds** automatically
- On refresh: sidebar stats update, table re-renders, selected row is preserved if it still exists
- If the selected record's `image_path` was `None` and is now populated → detail panel updates automatically
- Use `self.after(5000, self._schedule_auto_refresh)` pattern (not `threading.Timer`) — always runs on main thread

**Row preservation on refresh:**
- Store `self.selected_unique_id` when a row is selected
- After re-rendering the table, re-select the row matching that `unique_id` if it exists
- If it was deleted/filtered out, call `detail_panel.clear()`

---

## 🎨 Visual Design Rules

Use CustomTkinter defaults with these overrides:

**Colors (use CTk color tuples `[light, dark]`):**
- Selected row background: `["#E6F1FB", "#1a2a3a"]`
- Badge green: `["#EAF3DE", "#1a2e1a"]` text `["#3B6D11", "#9FE1CB"]`
- Badge amber: `["#FAEEDA", "#2e2010"]` text `["#854F0B", "#FAC775"]`
- Badge gray: `["#F1EFE8", "#2a2a28"]` text `["#5F5E5A", "#D3D1C7"]`

**Typography:**
- Unique IDs: `CTkFont(family="Courier New", size=11)` — monospace
- Section labels: 11px, muted, uppercase
- Weights: 13px, bold
- Body: 12px regular

**Spacing:**
- Panel padding: 12px
- Row height: ~44px
- Gap between action buttons: 6px

---

## 🧪 Empty & Error States

| State | What to show |
|---|---|
| Database not found | Error dialog: `"Database not found at runs/jewellery_detections.db. Is the detection system running?"` |
| No records at all | Table shows centered message: `"No detections yet. Start the detection system."` |
| Filter returns 0 results | `"No records match this filter."` |
| Video file missing | Button still shows but on click: dialog `"Video file not found: {path}"` |
| Image processing | Image button grayed out with tooltip `"Image is still being extracted"` |

---

## ✅ Acceptance Criteria

- [ ] `python viewer/app.py` launches a working window
- [ ] Clicking any row populates the detail panel with correct data
- [ ] `▶ Open video` opens the `.mp4` in the system video player
- [ ] `⊞ Open image` opens the `.jpg` in the system image viewer
- [ ] Disabled when file is not ready (grayed out, not clickable)
- [ ] `⎘ Copy unique ID` copies text to clipboard
- [ ] Table auto-refreshes every 5 seconds without losing selected row
- [ ] Sidebar nav filters work correctly
- [ ] Search filters by unique_id and weight in real time
- [ ] Works on both Linux/Raspberry Pi (`xdg-open`) and Windows (`os.startfile`)
- [ ] Thumbnails load without freezing the UI (background thread)
- [ ] Dark mode works correctly (tested with `set_appearance_mode("Dark")`)

---

## 🚫 Out of Scope

- No editing of database records
- No deletion of records or files
- No export to CSV or PDF
- No login / authentication
- No charting or analytics view
- No remote/network DB connection

---

## 💬 Notes for Claude Opus

- `customtkinter.CTkScrollableFrame` is the correct widget for the table — do NOT use `tkinter.Treeview` (ugly, hard to customize)
- Each "row" in the table is a `CTkFrame` containing `CTkLabel` widgets arranged in columns — build rows manually, not with a grid widget
- For column alignment, use `grid()` with fixed `minsize` on column configs inside each row frame
- `CTkImage` requires `light_image` and `dark_image` — pass the same PIL image for both if you only have one version
- Thumbnail loading MUST use `self.after(0, lambda: label.configure(image=ctk_img))` to update UI from a background thread — never update CTk widgets directly from non-main threads
- The DB path `runs/jewellery_detections.db` is relative — resolve it with `Path(__file__).parent.parent / "runs/jewellery_detections.db"` so it works regardless of working directory
- Use `customtkinter.set_appearance_mode("System")` so it respects the OS dark/light mode automatically

---

*End of PRD — Ready for Claude Opus implementation*
