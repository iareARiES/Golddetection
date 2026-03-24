import customtkinter


class Sidebar(customtkinter.CTkFrame):
    """Left navigation panel with filter items and stats."""

    NAV_ITEMS = [
        ("All Detections", "all"),
        ("Today", "today"),
        ("Pending", "pending"),
        ("Done", "done"),
        ("Partial", "partial"),
        ("Failed", "failed"),
        ("With Weight", "with_weight"),
    ]

    def __init__(self, parent, on_nav_change: callable, **kwargs):
        width = kwargs.pop("width", 200)
        super().__init__(parent, width=width, corner_radius=0, **kwargs)
        self.on_nav_change = on_nav_change
        self.active_filter = "all"
        self.nav_buttons = {}

        self.grid_propagate(False)

        title = customtkinter.CTkLabel(
            self, text="Gold Detection",
            font=customtkinter.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        title.grid(row=0, column=0, padx=16, pady=(20, 0), sticky="w")

        subtitle = customtkinter.CTkLabel(
            self, text="Viewer",
            font=customtkinter.CTkFont(size=11),
            text_color=("gray50", "gray60"),
            anchor="w"
        )
        subtitle.grid(row=1, column=0, padx=16, pady=(0, 20), sticky="w")

        for i, (label, mode) in enumerate(self.NAV_ITEMS):
            btn = customtkinter.CTkButton(
                self,
                text=f"  {label}",
                font=customtkinter.CTkFont(size=12),
                anchor="w",
                height=36,
                corner_radius=6,
                fg_color="transparent",
                text_color=("gray20", "gray80"),
                hover_color=("gray85", "gray25"),
                command=lambda m=mode: self._on_click(m),
            )
            btn.grid(row=i + 2, column=0, padx=8, pady=2, sticky="ew")
            self.nav_buttons[mode] = btn

        self.grid_columnconfigure(0, weight=1)

        sep = customtkinter.CTkFrame(self, height=1, fg_color=("gray75", "gray30"))
        sep.grid(row=len(self.NAV_ITEMS) + 3, column=0, padx=12, pady=16, sticky="ew")

        # Stats
        self.stat_total = customtkinter.CTkLabel(
            self, text="Total: 0", font=customtkinter.CTkFont(size=11),
            text_color=("gray40", "gray60"), anchor="w"
        )
        self.stat_total.grid(row=len(self.NAV_ITEMS) + 4, column=0, padx=16, sticky="w")

        self.stat_pending = customtkinter.CTkLabel(
            self, text="Pending: 0", font=customtkinter.CTkFont(size=11),
            text_color=("gray40", "gray60"), anchor="w"
        )
        self.stat_pending.grid(row=len(self.NAV_ITEMS) + 5, column=0, padx=16, pady=(2, 0), sticky="w")

        self.stat_done = customtkinter.CTkLabel(
            self, text="Done: 0", font=customtkinter.CTkFont(size=11),
            text_color=("gray40", "gray60"), anchor="w"
        )
        self.stat_done.grid(row=len(self.NAV_ITEMS) + 6, column=0, padx=16, pady=(2, 0), sticky="w")

        self.stat_failed = customtkinter.CTkLabel(
            self, text="Failed: 0", font=customtkinter.CTkFont(size=11),
            text_color=("gray40", "gray60"), anchor="w"
        )
        self.stat_failed.grid(row=len(self.NAV_ITEMS) + 7, column=0, padx=16, pady=(2, 0), sticky="w")

        self.stat_today = customtkinter.CTkLabel(
            self, text="Today: 0", font=customtkinter.CTkFont(size=11),
            text_color=("gray40", "gray60"), anchor="w"
        )
        self.stat_today.grid(row=len(self.NAV_ITEMS) + 8, column=0, padx=16, pady=(2, 0), sticky="w")

        self._highlight("all")

    def _on_click(self, mode: str):
        self.active_filter = mode
        self._highlight(mode)
        self.on_nav_change(mode)

    def _highlight(self, active_mode: str):
        for mode, btn in self.nav_buttons.items():
            if mode == active_mode:
                btn.configure(
                    fg_color=("#E6F1FB", "#1a2a3a"),
                    text_color=("#1a6dba", "#7ec4ff"),
                    font=customtkinter.CTkFont(size=12, weight="bold"),
                    border_width=0,
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=("gray20", "gray80"),
                    font=customtkinter.CTkFont(size=12),
                    border_width=0,
                )

    def update_stats(self, stats: dict):
        self.stat_total.configure(text=f"Total: {stats.get('total', 0)}")
        self.stat_pending.configure(text=f"Pending: {stats.get('pending', 0)}")
        self.stat_done.configure(text=f"Done: {stats.get('done', 0)}")
        self.stat_failed.configure(text=f"Failed: {stats.get('failed', 0)}")
        self.stat_today.configure(text=f"Today: {stats.get('today_count', 0)}")
