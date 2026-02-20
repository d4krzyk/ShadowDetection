from tkinter import ttk
import tkinter as tk


class UiMixin:
    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="nsw")

        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        params_frame = ttk.LabelFrame(left, text="Parametry", padding=8)
        params_frame.grid(row=0, column=0, sticky="ew")

        r = 0
        r = self._add_scale(
            params_frame,
            r,
            "v_thresh (0-255)",
            self.var_v_thresh,
            0,
            255,
            desc="Wyżej: więcej pikseli uznanych za cień (bardziej agresywnie).",
        )
        r = self._add_scale(
            params_frame,
            r,
            "hue_diff (0-40)",
            self.var_hue_diff,
            0,
            40,
            desc="Niżej: ostrzej odrzuca obiekty o innej barwie; wyżej: toleruje barwę.",
        )
        r = self._add_scale(
            params_frame,
            r,
            "min_area (0-5000)",
            self.var_min_area,
            0,
            5000,
            desc="Wyżej: usuwa drobny szum; niżej: zachowuje małe cienie.",
        )
        r = self._add_scale(
            params_frame,
            r,
            "min_elong*100 (100-400)",
            self.var_min_elong_x100,
            100,
            400,
            desc="Wyżej: przepuszcza tylko bardziej wydłużone cienie.",
        )
        r = self._add_scale(
            params_frame,
            r,
            "morph_kernel (3-31 odd)",
            self.var_morph_kernel,
            3,
            31,
            desc="Wyżej: mocniejsze domykanie/otwieranie; niżej: delikatniejsze.",
        )
        r = self._add_scale(
            params_frame,
            r,
            "soft_thresh*100 (5-60)",
            self.var_soft_shadow_thresh_x100,
            5,
            60,
            desc="Niżej: lapie slabsze, rozproszone cienie.",
        )
        r = self._add_scale(
            params_frame,
            r,
            "soft_scale*100 (10-200)",
            self.var_soft_shadow_scale_x100,
            10,
            200,
            desc="Wyżej: mniej agresywne uznawanie ciemnosci za cien.",
        )

        opts_frame = ttk.LabelFrame(left, text="Opcje", padding=8)
        opts_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self._opts_frame = opts_frame

        ttk.Checkbutton(opts_frame, text="GEOM (walidacja geometryczna)", variable=self.var_use_geom, command=self._schedule_render).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(opts_frame, text="CLAHE (lokalny kontrast)", variable=self.var_use_clahe, command=self._schedule_render).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(opts_frame, text="Overlay (maska jako nakladka)", variable=self.var_overlay, command=self._schedule_render).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(opts_frame, text="Soft shadow (grayscale)", variable=self.var_soft_shadow, command=self._schedule_render).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(opts_frame, text="Pokaz soft mask", variable=self.var_show_soft_mask, command=self._schedule_render).grid(row=4, column=0, sticky="w")
        ttk.Checkbutton(opts_frame, text="Dir: Hough", variable=self.var_use_dir_hough, command=self._schedule_render).grid(row=5, column=0, sticky="w")
        ttk.Checkbutton(opts_frame, text="Dir: PCA", variable=self.var_use_dir_pca, command=self._schedule_render).grid(row=6, column=0, sticky="w")
        ttk.Checkbutton(opts_frame, text="Dir: Gradient", variable=self.var_use_dir_grad, command=self._schedule_render).grid(row=7, column=0, sticky="w")

        btns = ttk.Frame(left)
        btns.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(btns, text="Zaladuj dane", command=self._load_new_source).grid(row=0, column=0, sticky="ew")
        self.random_btn = ttk.Button(btns, text="Losuj (r)", command=self._random_image)
        self.random_btn.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(btns, text="Zapisz maskę (s)", command=self._save_outputs).grid(row=0, column=2, sticky="ew", padx=(8, 0))
        self.gif_pause_btn = ttk.Button(btns, text="Pauza (p)", command=self._toggle_gif_pause, state="disabled")
        self.gif_pause_btn.grid(row=0, column=3, sticky="ew", padx=(8, 0))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        btns.columnconfigure(2, weight=1)
        btns.columnconfigure(3, weight=1)

        self.status_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.status_var, wraplength=320, foreground="#444").grid(row=3, column=0, sticky="w", pady=(8, 0))

        self.preview_label = ttk.Label(right)
        self.preview_label.grid(row=0, column=0, sticky="nsew")

    def _add_scale(self, parent, row, label, var, frm, to, desc=None):
        box = ttk.Frame(parent)
        box.grid(row=row, column=0, sticky="ew", pady=2)
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text=label).grid(row=0, column=0, sticky="w")
        scale = ttk.Scale(box, from_=frm, to=to, orient="horizontal", variable=var, command=lambda _v: self._on_scale(var))
        scale.grid(row=0, column=1, sticky="ew", padx=(8, 8))

        scale.set(var.get())

        value_lbl = ttk.Label(box, width=5, anchor="e")
        value_lbl.grid(row=0, column=2, sticky="e")
        value_lbl.configure(text=str(var.get()))

        def update_label(*_):
            value_lbl.configure(text=str(int(var.get())))

        var.trace_add("write", update_label)

        if desc:
            ttk.Label(box, text=desc, foreground="#666").grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 0))
            return row + 2
        return row + 1

    def _on_scale(self, var):
        var.set(int(float(var.get())))
        self._schedule_render()

    def _bind_events(self):
        self.bind("<KeyPress-r>", lambda _e: self._random_image())
        self.bind("<KeyPress-s>", lambda _e: self._save_outputs())
        self.bind("<KeyPress-p>", lambda _e: self._toggle_gif_pause())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.bind("<KeyPress-q>", lambda _e: self.destroy())
