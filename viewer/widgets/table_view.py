import customtkinter
import threading
from pathlib import Path
from datetime import datetime

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from viewer.file_opener import open_file


class TableView(customtkinter.CTkScrollableFrame):
    """Scrollable table of detection records with thumbnails and action buttons."""

    # Column widths
    COL_THUMB = 60
    COL_ID = 200
    COL_DATE = 150
    COL_WEIGHT = 80
    COL_VIDEO = 70
    COL_IMAGE = 80
    COL_ACTIONS = 140

    # Colors
    ROW_SELECTED = ("#E6F1FB", "#1a2a3a")
    ROW_HOVER = ("#F0F0F0", "#2a2a2a")
    ROW_NORMAL = ("transparent", "transparent")
    BADGE_GREEN_BG = ("#EAF3DE", "#1a2e1a")
    BADGE_GREEN_FG = ("#3B6D11", "#9FE1CB")
    BADGE_AMBER_BG = ("#FAEEDA", "#2e2010")
    BADGE_AMBER_FG = ("#854F0B", "#FAC775")
    BADGE_RED_BG = ("#FDECEC", "#2e1a1a")
    BADGE_RED_FG = ("#B91C1C", "#FCA5A5")
    BADGE_GRAY_BG = ("#F1EFE8", "#2a2a28")
    BADGE_GRAY_FG = ("#5F5E5A", "#D3D1C7")

    def __init__(self, parent, on_row_select: callable, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_row_select = on_row_select
        self.rows = []
        self.selected_id = None
        self._thumbnail_cache = {}

        # Header
        self._create_header()

    def _create_header(self):
        header = customtkinter.CTkFrame(self, height=32, fg_color=("gray90", "gray17"), corner_radius=0)
        header.pack(fill="x", padx=0, pady=(0, 4))
        header.grid_columnconfigure(1, weight=1)

        cols = [
            ("", self.COL_THUMB),
            ("Unique ID", self.COL_ID),
            ("Date & Time", self.COL_DATE),
            ("Weight", self.COL_WEIGHT),
            ("Video", self.COL_VIDEO),
            ("Image", self.COL_IMAGE),
            ("Actions", self.COL_ACTIONS),
        ]
        for i, (text, width) in enumerate(cols):
            lbl = customtkinter.CTkLabel(
                header, text=text, width=width,
                font=customtkinter.CTkFont(size=11, weight="bold"),
                text_color=("gray40", "gray60"),
                anchor="w"
            )
            lbl.grid(row=0, column=i, padx=(8 if i == 0 else 4, 4), pady=6, sticky="w")

    def load_data(self, records: list[dict]):
        """Clear and re-render all rows from records list."""
        # Clear existing rows
        for row_frame in self.rows:
            row_frame.destroy()
        self.rows.clear()

        if not records:
            empty = customtkinter.CTkLabel(
                self, text="No detections found.",
                font=customtkinter.CTkFont(size=13),
                text_color=("gray50", "gray60"),
            )
            empty.pack(pady=40)
            self.rows.append(empty)
            return

        for record in records:
            row = self._create_row(record)
            row.pack(fill="x", padx=0, pady=1)
            self.rows.append(row)

    def _create_row(self, record: dict) -> customtkinter.CTkFrame:
        is_selected = record.get("id") == self.selected_id
        row_color = self.ROW_SELECTED if is_selected else self.ROW_NORMAL

        row = customtkinter.CTkFrame(self, height=44, fg_color=row_color, corner_radius=4)
        row.grid_columnconfigure(1, weight=1)
        row._record = record  # stash for click handler

        # Bind clicks on the row frame
        row.bind("<Button-1>", lambda e, r=record: self._select_row(r))
        row.bind("<Double-Button-1>", lambda e, r=record: self._double_click(r))
        row.bind("<Enter>", lambda e, f=row: self._on_hover(f, True))
        row.bind("<Leave>", lambda e, f=row: self._on_hover(f, False))

        # --- Thumbnail ---
        thumb_label = customtkinter.CTkLabel(row, text="?", width=self.COL_THUMB, height=40,
                                              fg_color=self.BADGE_GRAY_BG, corner_radius=4,
                                              font=customtkinter.CTkFont(size=14),
                                              text_color=self.BADGE_GRAY_FG)
        thumb_label.grid(row=0, column=0, padx=(8, 4), pady=2, sticky="w")
        thumb_label.bind("<Button-1>", lambda e, r=record: self._select_row(r))

        image_path = record.get("image_path")
        if image_path and HAS_PIL:
            self._load_thumbnail_async(image_path, thumb_label)

        # --- Unique ID ---
        uid = record.get("unique_id", "—")
        display_uid = uid[:24] + "…" if len(uid) > 24 else uid
        id_label = customtkinter.CTkLabel(row, text=display_uid, width=self.COL_ID,
                                           font=customtkinter.CTkFont(family="Courier New", size=11),
                                           anchor="w")
        id_label.grid(row=0, column=1, padx=4, pady=2, sticky="w")
        id_label.bind("<Button-1>", lambda e, r=record: self._select_row(r))

        # --- Date & Time ---
        captured = record.get("captured_at", "")
        try:
            dt = datetime.fromisoformat(captured)
            date_str = dt.strftime("%d %b %Y, %H:%M:%S")
        except (ValueError, TypeError):
            date_str = captured or "—"
        date_label = customtkinter.CTkLabel(row, text=date_str, width=self.COL_DATE,
                                             font=customtkinter.CTkFont(size=12), anchor="w")
        date_label.grid(row=0, column=2, padx=4, pady=2, sticky="w")
        date_label.bind("<Button-1>", lambda e, r=record: self._select_row(r))

        # --- Weight ---
        weight = record.get("weight_grams", "unknown")
        if weight == "unknown":
            w_text, w_bg, w_fg = "unknown", self.BADGE_GRAY_BG, self.BADGE_GRAY_FG
        else:
            w_text, w_bg, w_fg = f"{weight} g", self.BADGE_GREEN_BG, self.BADGE_GREEN_FG
        weight_badge = customtkinter.CTkLabel(row, text=w_text, width=self.COL_WEIGHT,
                                               fg_color=w_bg, text_color=w_fg,
                                               corner_radius=4,
                                               font=customtkinter.CTkFont(size=11, weight="bold"))
        weight_badge.grid(row=0, column=3, padx=4, pady=6, sticky="w")
        weight_badge.bind("<Button-1>", lambda e, r=record: self._select_row(r))

        # --- Video status ---
        video_path = record.get("video_path", "")
        if video_path and Path(video_path).exists():
            v_text, v_bg, v_fg = "✓ saved", self.BADGE_GREEN_BG, self.BADGE_GREEN_FG
        else:
            v_text, v_bg, v_fg = "✗ missing", self.BADGE_RED_BG, self.BADGE_RED_FG
        vid_badge = customtkinter.CTkLabel(row, text=v_text, width=self.COL_VIDEO,
                                            fg_color=v_bg, text_color=v_fg,
                                            corner_radius=4,
                                            font=customtkinter.CTkFont(size=10))
        vid_badge.grid(row=0, column=4, padx=4, pady=6, sticky="w")
        vid_badge.bind("<Button-1>", lambda e, r=record: self._select_row(r))

        # --- Image status ---
        img_path = record.get("image_path")
        if img_path and Path(img_path).exists():
            i_text, i_bg, i_fg = "✓ ready", self.BADGE_GREEN_BG, self.BADGE_GREEN_FG
        elif img_path:
            i_text, i_bg, i_fg = "✗ missing", self.BADGE_RED_BG, self.BADGE_RED_FG
        else:
            i_text, i_bg, i_fg = "⏳ pending", self.BADGE_AMBER_BG, self.BADGE_AMBER_FG
        img_badge = customtkinter.CTkLabel(row, text=i_text, width=self.COL_IMAGE,
                                            fg_color=i_bg, text_color=i_fg,
                                            corner_radius=4,
                                            font=customtkinter.CTkFont(size=10))
        img_badge.grid(row=0, column=5, padx=4, pady=6, sticky="w")
        img_badge.bind("<Button-1>", lambda e, r=record: self._select_row(r))

        # --- Action buttons ---
        actions = customtkinter.CTkFrame(row, fg_color="transparent", width=self.COL_ACTIONS)
        actions.grid(row=0, column=6, padx=4, pady=4, sticky="w")

        vid_btn = customtkinter.CTkButton(
            actions, text="▶", width=40, height=28, corner_radius=6,
            font=customtkinter.CTkFont(size=12),
            command=lambda: open_file(record.get("video_path", "")),
        )
        vid_btn.pack(side="left", padx=(0, 4))

        img_btn_state = "normal" if (img_path and Path(img_path).exists()) else "disabled"
        img_btn = customtkinter.CTkButton(
            actions, text="⊞", width=40, height=28, corner_radius=6,
            font=customtkinter.CTkFont(size=12),
            fg_color=("gray70", "gray30") if img_btn_state == "disabled" else None,
            state=img_btn_state,
            command=lambda: open_file(record.get("image_path", "")),
        )
        img_btn.pack(side="left")

        return row

    def _select_row(self, record: dict):
        self.selected_id = record.get("id")
        # Update all row backgrounds
        for row_frame in self.rows:
            if hasattr(row_frame, "_record"):
                if row_frame._record.get("id") == self.selected_id:
                    row_frame.configure(fg_color=self.ROW_SELECTED)
                else:
                    row_frame.configure(fg_color=self.ROW_NORMAL)
        self.on_row_select(record)

    def _double_click(self, record: dict):
        img = record.get("image_path")
        vid = record.get("video_path")
        if img and Path(img).exists():
            open_file(img)
        elif vid:
            open_file(vid)

    def _on_hover(self, frame, entering: bool):
        if hasattr(frame, "_record") and frame._record.get("id") != self.selected_id:
            frame.configure(fg_color=self.ROW_HOVER if entering else self.ROW_NORMAL)

    def clear_selection(self):
        self.selected_id = None
        for row_frame in self.rows:
            if hasattr(row_frame, "_record"):
                row_frame.configure(fg_color=self.ROW_NORMAL)

    def _load_thumbnail_async(self, image_path: str, label_widget):
        """Load image in thread, update label on main thread via after()."""
        if image_path in self._thumbnail_cache:
            ctk_img = self._thumbnail_cache[image_path]
            label_widget.configure(image=ctk_img, text="")
            return

        def _load():
            try:
                if not Path(image_path).exists():
                    return
                pil_img = Image.open(image_path)
                # Resize to fit 60x40 maintaining aspect ratio
                pil_img.thumbnail((56, 40), Image.LANCZOS)
                ctk_img = customtkinter.CTkImage(
                    light_image=pil_img, dark_image=pil_img,
                    size=(pil_img.width, pil_img.height)
                )
                self._thumbnail_cache[image_path] = ctk_img
                # Update on main thread
                self.after(0, lambda: label_widget.configure(image=ctk_img, text=""))
            except Exception:
                pass

        threading.Thread(target=_load, daemon=True).start()
