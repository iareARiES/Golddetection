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
    """Scrollable table showing detection events with status badges."""

    COL_THUMB  = 60
    COL_DATE   = 150
    COL_WEIGHT = 80
    COL_STATUS = 90
    COL_C270V  = 60
    COL_LENV   = 60
    COL_DUR    = 60
    COL_ACT    = 50

    ROW_SELECTED = ("#E6F1FB", "#1a2a3a")
    ROW_HOVER    = ("#F0F0F0", "#2a2a2a")
    ROW_NORMAL   = "transparent"

    BADGE = {
        "pending":    (("#FAEEDA", "#2e2010"), ("#854F0B", "#FAC775")),
        "processing": (("#E6F1FB", "#1a2a3a"), ("#1a6dba", "#7ec4ff")),
        "done":       (("#EAF3DE", "#1a2e1a"), ("#3B6D11", "#9FE1CB")),
        "partial":    (("#FFF8E1", "#2e2a10"), ("#7C5C00", "#FFD54F")),
        "failed":     (("#FDECEC", "#2e1a1a"), ("#B91C1C", "#FCA5A5")),
    }
    BADGE_GRAY = (("#F1EFE8", "#2a2a28"), ("#5F5E5A", "#D3D1C7"))
    BADGE_GREEN = (("#EAF3DE", "#1a2e1a"), ("#3B6D11", "#9FE1CB"))
    BADGE_RED   = (("#FDECEC", "#2e1a1a"), ("#B91C1C", "#FCA5A5"))

    def __init__(self, parent, on_row_select: callable, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_row_select = on_row_select
        self.rows = []
        self.selected_id = None
        self._thumbnail_cache = {}
        self._create_header()

    def _create_header(self):
        hdr = customtkinter.CTkFrame(self, height=32, fg_color=("gray90", "gray17"), corner_radius=0)
        hdr.pack(fill="x", padx=0, pady=(0, 4))
        hdr.grid_columnconfigure(1, weight=1)

        cols = [
            ("", self.COL_THUMB), ("Date/Time", self.COL_DATE),
            ("Status", self.COL_STATUS), ("Weight", self.COL_WEIGHT),
            ("Dur", self.COL_DUR), ("C270", self.COL_C270V),
            ("Lenovo", self.COL_LENV), ("", self.COL_ACT),
        ]
        for i, (text, width) in enumerate(cols):
            lbl = customtkinter.CTkLabel(
                hdr, text=text, width=width,
                font=customtkinter.CTkFont(size=11, weight="bold"),
                text_color=("gray40", "gray60"), anchor="w",
            )
            lbl.grid(row=0, column=i, padx=(8 if i == 0 else 4, 4), pady=6, sticky="w")

    def load_data(self, records: list[dict]):
        for rf in self.rows:
            rf.destroy()
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

        for rec in records:
            row = self._create_row(rec)
            row.pack(fill="x", padx=0, pady=1)
            self.rows.append(row)

    def _badge(self, parent, text, bg, fg, width):
        return customtkinter.CTkLabel(
            parent, text=text, width=width,
            fg_color=bg, text_color=fg, corner_radius=4,
            font=customtkinter.CTkFont(size=10, weight="bold"),
        )

    def _file_badge(self, parent, path, width):
        if path and Path(path).exists():
            return self._badge(parent, "OK", *self.BADGE_GREEN, width)
        elif path:
            return self._badge(parent, "MISS", *self.BADGE_RED, width)
        return self._badge(parent, "-", *self.BADGE_GRAY, width)

    def _create_row(self, rec: dict) -> customtkinter.CTkFrame:
        is_sel = rec.get("id") == self.selected_id
        row = customtkinter.CTkFrame(self, height=44,
            fg_color=self.ROW_SELECTED if is_sel else self.ROW_NORMAL, corner_radius=4)
        row.grid_columnconfigure(1, weight=1)
        row._record = rec
        row.bind("<Button-1>", lambda e, r=rec: self._select_row(r))
        row.bind("<Enter>", lambda e, f=row: self._hover(f, True))
        row.bind("<Leave>", lambda e, f=row: self._hover(f, False))

        # Thumbnail (C270 crop if available)
        thumb = customtkinter.CTkLabel(row, text="?", width=self.COL_THUMB, height=40,
            fg_color=self.BADGE_GRAY[0], corner_radius=4,
            font=customtkinter.CTkFont(size=14), text_color=self.BADGE_GRAY[1])
        thumb.grid(row=0, column=0, padx=(8, 4), pady=2, sticky="w")
        thumb.bind("<Button-1>", lambda e, r=rec: self._select_row(r))
        img_path = rec.get("image_c270")
        if img_path and HAS_PIL:
            self._load_thumb(img_path, thumb)

        # Date
        captured = rec.get("captured_at", "")
        try:
            dt = datetime.fromisoformat(captured)
            ds = dt.strftime("%d %b %Y, %H:%M")
        except (ValueError, TypeError):
            ds = captured or "-"
        dl = customtkinter.CTkLabel(row, text=ds, width=self.COL_DATE,
            font=customtkinter.CTkFont(size=12), anchor="w")
        dl.grid(row=0, column=1, padx=4, pady=2, sticky="w")
        dl.bind("<Button-1>", lambda e, r=rec: self._select_row(r))

        # Status
        status = rec.get("processing_status", "pending")
        s_bg, s_fg = self.BADGE.get(status, self.BADGE_GRAY)
        sl = self._badge(row, status, s_bg, s_fg, self.COL_STATUS)
        sl.grid(row=0, column=2, padx=4, pady=6, sticky="w")

        # Weight
        wt = rec.get("weight")
        if wt and wt not in ("None", "unavailable", ""):
            wl = self._badge(row, f"{wt} g", *self.BADGE_GREEN, self.COL_WEIGHT)
        else:
            wl = self._badge(row, "pending" if status != "done" else "none",
                             *self.BADGE_GRAY, self.COL_WEIGHT)
        wl.grid(row=0, column=3, padx=4, pady=6, sticky="w")

        # Duration
        dur = rec.get("duration_sec")
        dur_text = f"{int(dur)}s" if dur else "-"
        durl = customtkinter.CTkLabel(row, text=dur_text, width=self.COL_DUR,
            font=customtkinter.CTkFont(size=11), anchor="w")
        durl.grid(row=0, column=4, padx=4, pady=2, sticky="w")

        # C270 video
        c270v = self._file_badge(row, rec.get("c270_video_path"), self.COL_C270V)
        c270v.grid(row=0, column=5, padx=4, pady=6, sticky="w")

        # Lenovo video
        lenv = self._file_badge(row, rec.get("lenovo_video_path"), self.COL_LENV)
        lenv.grid(row=0, column=6, padx=4, pady=6, sticky="w")

        # Play button
        c270_path = rec.get("c270_video_path", "")
        btn_state = "normal" if (c270_path and Path(c270_path).exists()) else "disabled"
        btn = customtkinter.CTkButton(
            row, text="Play", width=44, height=28, corner_radius=6,
            font=customtkinter.CTkFont(size=11),
            fg_color=("gray70", "gray30") if btn_state == "disabled" else None,
            state=btn_state,
            command=lambda p=c270_path: open_file(p),
        )
        btn.grid(row=0, column=7, padx=4, pady=4, sticky="w")

        return row

    def _select_row(self, rec: dict):
        self.selected_id = rec.get("id")
        for rf in self.rows:
            if hasattr(rf, "_record"):
                rf.configure(fg_color=self.ROW_SELECTED
                    if rf._record.get("id") == self.selected_id else self.ROW_NORMAL)
        self.on_row_select(rec)

    def _hover(self, frame, entering: bool):
        if hasattr(frame, "_record") and frame._record.get("id") != self.selected_id:
            frame.configure(fg_color=self.ROW_HOVER if entering else self.ROW_NORMAL)

    def _load_thumb(self, image_path, label):
        if image_path in self._thumbnail_cache:
            label.configure(image=self._thumbnail_cache[image_path], text="")
            return
        def _bg():
            try:
                if not Path(image_path).exists(): return
                img = Image.open(image_path)
                img.thumbnail((56, 40), Image.LANCZOS)
                ctk = customtkinter.CTkImage(light_image=img, dark_image=img,
                    size=(img.width, img.height))
                self._thumbnail_cache[image_path] = ctk
                if label.winfo_exists():
                    self.after(0, lambda: label.configure(image=ctk, text=""))
            except Exception:
                pass
        threading.Thread(target=_bg, daemon=True).start()
