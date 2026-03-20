import customtkinter


class Topbar(customtkinter.CTkFrame):
    """Horizontal bar with section title, search box, and refresh button."""

    def __init__(self, parent, on_search: callable, on_refresh: callable, **kwargs):
        super().__init__(parent, height=50, corner_radius=0, **kwargs)
        self.on_search = on_search
        self.on_refresh = on_refresh
        self._search_after_id = None

        self.grid_columnconfigure(1, weight=1)

        # --- Section title ---
        self.title_label = customtkinter.CTkLabel(
            self, text="All detections",
            font=customtkinter.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        self.title_label.grid(row=0, column=0, padx=(16, 12), pady=12, sticky="w")

        # --- Search box ---
        self.search_var = customtkinter.StringVar()
        self.search_entry = customtkinter.CTkEntry(
            self,
            placeholder_text="Search by ID, weight, or notes...",
            textvariable=self.search_var,
            width=280,
            height=32,
            corner_radius=8,
        )
        self.search_entry.grid(row=0, column=1, padx=8, pady=12, sticky="e")
        self.search_var.trace_add("write", self._on_search_change)

        # --- Refresh button ---
        self.refresh_btn = customtkinter.CTkButton(
            self, text="⟳ Refresh", width=90, height=32,
            corner_radius=8,
            fg_color="transparent",
            border_width=1,
            border_color=("gray60", "gray40"),
            text_color=("gray20", "gray80"),
            hover_color=("gray85", "gray25"),
            command=self.on_refresh,
        )
        self.refresh_btn.grid(row=0, column=2, padx=(4, 16), pady=12, sticky="e")

    def _on_search_change(self, *args):
        """Debounced search — waits 300ms after last keystroke."""
        if self._search_after_id is not None:
            self.after_cancel(self._search_after_id)
        self._search_after_id = self.after(300, self._fire_search)

    def _fire_search(self):
        self._search_after_id = None
        self.on_search(self.search_var.get())

    def set_title(self, title: str):
        self.title_label.configure(text=title)
