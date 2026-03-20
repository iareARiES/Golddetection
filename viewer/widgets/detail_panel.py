import customtkinter
import threading
from pathlib import Path
from datetime import datetime

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from viewer.file_opener import open_file, reveal_in_folder, copy_to_clipboard


class DetailPanel(customtkinter.CTkFrame):
    """Right panel showing detail for the selected detection."""

    def __init__(self, parent, **kwargs):
        width = kwargs.pop("width", 280)
        super().__init__(parent, width=width, corner_radius=0, **kwargs)
        self.grid_propagate(False)
        self.grid_columnconfigure(0, weight=1)

        self._current_record = None
        self._image_ref = None  # keep reference to prevent GC

        # --- Empty state ---
        self.empty_label = customtkinter.CTkLabel(
            self, text="Select a detection\nto view details",
            font=customtkinter.CTkFont(size=13),
            text_color=("gray50", "gray60"),
            justify="center",
        )
        self.empty_label.grid(row=0, column=0, pady=120)

        # --- Content frame (hidden initially) ---
        self.content = customtkinter.CTkFrame(self, fg_color="transparent")

        # Image preview
        self.image_label = customtkinter.CTkLabel(
            self.content, text="Image processing...",
            width=256, height=160,
            fg_color=("#F1EFE8", "#2a2a28"),
            corner_radius=8,
            font=customtkinter.CTkFont(size=12),
            text_color=("gray50", "gray60"),
        )
        self.image_label.pack(padx=12, pady=(12, 4), fill="x")

        self.image_type_label = customtkinter.CTkLabel(
            self.content, text="",
            font=customtkinter.CTkFont(size=10),
            text_color=("gray50", "gray60"),
        )
        self.image_type_label.pack(padx=12, pady=(0, 8))

        # Separator
        sep1 = customtkinter.CTkFrame(self.content, height=1, fg_color=("gray80", "gray30"))
        sep1.pack(fill="x", padx=12, pady=4)

        # Details section
        self.details_frame = customtkinter.CTkFrame(self.content, fg_color="transparent")
        self.details_frame.pack(fill="x", padx=12, pady=8)

        self._detail_labels = {}
        detail_fields = ["Unique ID", "Weight", "Captured", "Date", "Status"]
        for i, field in enumerate(detail_fields):
            key_lbl = customtkinter.CTkLabel(
                self.details_frame, text=field.upper(),
                font=customtkinter.CTkFont(size=10),
                text_color=("gray50", "gray60"),
                anchor="w",
            )
            key_lbl.grid(row=i, column=0, padx=(0, 8), pady=2, sticky="w")

            val_lbl = customtkinter.CTkLabel(
                self.details_frame, text="—",
                font=customtkinter.CTkFont(size=12, weight="bold" if field == "Weight" else "normal",
                                            family="Courier New" if field == "Unique ID" else None),
                anchor="w",
            )
            val_lbl.grid(row=i, column=1, pady=2, sticky="w")
            self._detail_labels[field] = val_lbl

        self.details_frame.grid_columnconfigure(1, weight=1)

        # Separator
        sep2 = customtkinter.CTkFrame(self.content, height=1, fg_color=("gray80", "gray30"))
        sep2.pack(fill="x", padx=12, pady=8)

        # Action buttons
        self.btn_video = customtkinter.CTkButton(
            self.content, text="▶  Open Video", height=34, corner_radius=8,
            command=self._open_video,
        )
        self.btn_video.pack(fill="x", padx=12, pady=(4, 3))

        self.btn_image = customtkinter.CTkButton(
            self.content, text="⊞  Open Image", height=34, corner_radius=8,
            fg_color=("gray70", "gray30"),
            command=self._open_image,
        )
        self.btn_image.pack(fill="x", padx=12, pady=3)

        self.btn_copy = customtkinter.CTkButton(
            self.content, text="⎘  Copy Unique ID", height=34, corner_radius=8,
            fg_color="transparent",
            border_width=1, border_color=("gray60", "gray40"),
            text_color=("gray20", "gray80"),
            hover_color=("gray85", "gray25"),
            command=self._copy_id,
        )
        self.btn_copy.pack(fill="x", padx=12, pady=3)

        self.btn_folder = customtkinter.CTkButton(
            self.content, text="📁  Reveal in Folder", height=34, corner_radius=8,
            fg_color="transparent",
            border_width=1, border_color=("gray60", "gray40"),
            text_color=("gray20", "gray80"),
            hover_color=("gray85", "gray25"),
            command=self._reveal,
        )
        self.btn_folder.pack(fill="x", padx=12, pady=(3, 12))

    def load_record(self, record: dict):
        """Populate all fields and image for this record."""
        self._current_record = record
        self.empty_label.grid_forget()
        self.content.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)

        # --- Image ---
        img_path = record.get("image_path")
        if img_path and Path(img_path).exists() and HAS_PIL:
            self.image_type_label.configure(text="best frame extracted")
            self._load_image_async(img_path)
        else:
            self.image_label.configure(image=None, text="Image processing...")
            self._image_ref = None
            self.image_type_label.configure(text="awaiting extraction")

        # --- Details ---
        uid = record.get("unique_id", "—")
        display_uid = uid[:20] + "…" if len(uid) > 20 else uid
        self._detail_labels["Unique ID"].configure(text=display_uid)

        weight = record.get("weight_grams", "unknown")
        self._detail_labels["Weight"].configure(text=f"{weight} g" if weight != "unknown" else "unknown")

        captured = record.get("captured_at", "")
        try:
            dt = datetime.fromisoformat(captured)
            self._detail_labels["Captured"].configure(text=dt.strftime("%H:%M:%S"))
            self._detail_labels["Date"].configure(text=dt.strftime("%d %b %Y"))
        except (ValueError, TypeError):
            self._detail_labels["Captured"].configure(text="—")
            self._detail_labels["Date"].configure(text="—")

        if record.get("image_path"):
            self._detail_labels["Status"].configure(text="complete", text_color=("#3B6D11", "#9FE1CB"))
        else:
            self._detail_labels["Status"].configure(text="processing", text_color=("#854F0B", "#FAC775"))

        # --- Buttons ---
        self.btn_video.configure(state="normal")
        if img_path and Path(img_path).exists():
            self.btn_image.configure(state="normal", fg_color=None)
        else:
            self.btn_image.configure(state="disabled", fg_color=("gray70", "gray30"))

    def clear(self):
        """Reset to empty state."""
        self._current_record = None
        self._image_ref = None
        self.content.grid_forget()
        self.empty_label.grid(row=0, column=0, pady=120)

    def _load_image_async(self, image_path: str):
        def _load():
            try:
                pil_img = Image.open(image_path)
                pil_img.thumbnail((256, 160), Image.LANCZOS)
                ctk_img = customtkinter.CTkImage(
                    light_image=pil_img, dark_image=pil_img,
                    size=(pil_img.width, pil_img.height)
                )
                self._image_ref = ctk_img
                self.after(0, lambda: self.image_label.configure(image=ctk_img, text=""))
            except Exception:
                self.after(0, lambda: self.image_label.configure(image=None, text="Failed to load"))

        threading.Thread(target=_load, daemon=True).start()

    def _open_video(self):
        if self._current_record:
            path = self._current_record.get("video_path", "")
            if not open_file(path):
                self._show_error(f"Video file not found:\n{path}")

    def _open_image(self):
        if self._current_record:
            path = self._current_record.get("image_path", "")
            if not open_file(path):
                self._show_error(f"Image file not found:\n{path}")

    def _copy_id(self):
        if self._current_record:
            copy_to_clipboard(self.winfo_toplevel(), self._current_record.get("unique_id", ""))

    def _reveal(self):
        if self._current_record:
            path = self._current_record.get("video_path", "")
            reveal_in_folder(path)

    def _show_error(self, message: str):
        dialog = customtkinter.CTkToplevel(self)
        dialog.title("File Not Found")
        dialog.geometry("400x120")
        dialog.resizable(False, False)
        customtkinter.CTkLabel(dialog, text=message, wraplength=360).pack(pady=20)
        customtkinter.CTkButton(dialog, text="OK", width=80, command=dialog.destroy).pack()
