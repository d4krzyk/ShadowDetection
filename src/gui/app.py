import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog

import cv2

from src.load_data import get_random_image_from_dir
from src.shadow_detection import detect_shadow_physics
from .media_player import MediaPlayerMixin
from .renderer import RenderWorkerMixin
from .source_dialog import SourceDialogMixin
from .ui import UiMixin


def _clamp_int(v, lo, hi):
    return int(max(lo, min(hi, int(v))))


class ShadowTunerApp(tk.Tk, UiMixin, SourceDialogMixin, MediaPlayerMixin, RenderWorkerMixin):
    """GUI do strojenia detekcji cieni (Tkinter)."""

    def __init__(self, images_folder: str | None = None):
        super().__init__()

        self.title("ShadowDetection - Tuner (Tkinter)")
        self.minsize(1100, 520)

        self._clamp_int = _clamp_int

        self.withdraw()
        source = self._prompt_source()
        if source is None:
            self.destroy()
            return

        self._source_mode = source["mode"]
        self.images_folder = source.get("folder")
        self.img_path = source.get("path")
        self.img_bgr = None
        self.deiconify()

        self._gif_frames = []
        self._gif_durations = []
        self._gif_index = 0
        self._gif_playing = False
        self._gif_paused = False
        self._gif_job = None
        self._display_name = ""

        self._video_cap = None
        self._video_fps = 0.0
        self._video_frame_index = 0
        self._media_type = None

        self._render_lock = threading.Lock()
        self._render_req_id = 0
        self._worker_thread = None
        self._result_q: queue.Queue = queue.Queue()

        self._prev_light_vec = None

        self.VIEW_W = 800
        self.VIEW_H = 650
        self.COMPASS_W = 180

        self.var_v_thresh = tk.IntVar(value=120)
        self.var_hue_diff = tk.IntVar(value=20)
        self.var_min_area = tk.IntVar(value=250)
        self.var_min_elong_x100 = tk.IntVar(value=100)
        self.var_morph_kernel = tk.IntVar(value=5)
        self.var_soft_shadow = tk.BooleanVar(value=True)
        self.var_soft_shadow_thresh_x100 = tk.IntVar(value=30)
        self.var_soft_shadow_scale_x100 = tk.IntVar(value=20)
        self.var_show_soft_mask = tk.BooleanVar(value=True)

        self.var_use_dir_hough = tk.BooleanVar(value=True)
        self.var_use_dir_pca = tk.BooleanVar(value=True)
        self.var_use_dir_grad = tk.BooleanVar(value=True)

        self.var_use_geom = tk.BooleanVar(value=True)
        self.var_use_clahe = tk.BooleanVar(value=False)
        self.var_overlay = tk.BooleanVar(value=True)


        self._build_ui()
        self._bind_events()
        self.preview_label.bind("<Configure>", self._on_preview_resize)

        if self._source_mode == "folder" and self.images_folder:
            self.img_path = get_random_image_from_dir(self.images_folder, include_video=True)
        self._set_image_source(self.img_path)
        self._apply_source_mode_ui()

        self.status_var.set(f"Start: {self.img_path}")
        self._schedule_render(reason="init")
        self.after(50, self._poll_results)

    def _apply_source_mode_ui(self):
        if self._source_mode == "folder":
            self._opts_frame.grid()
            self.random_btn.grid()
            return
        self._opts_frame.grid()
        self.random_btn.grid_remove()

    def _load_new_source(self):
        self.withdraw()
        source = self._prompt_source()
        self.deiconify()
        if source is None:
            return
        self._source_mode = source["mode"]
        self.images_folder = source.get("folder")
        self.img_path = source.get("path")
        if self._source_mode == "folder" and self.images_folder:
            self.img_path = get_random_image_from_dir(self.images_folder, include_video=True)
        self._set_image_source(self.img_path)
        self._apply_source_mode_ui()
        self.status_var.set(f"Start: {self.img_path}")
        self._schedule_render(reason="load")

    def _random_image(self):
        if self._source_mode != "folder" or not self.images_folder:
            return
        try:
            self._set_image_source(get_random_image_from_dir(self.images_folder, include_video=True))
            self.status_var.set(f"Nowy obraz: {self.img_path}")
            self._schedule_render(reason="random")
        except Exception as e:
            self.status_var.set(f"Blad losowania: {e}")

    def _toggle_gif_pause(self):
        if not self._gif_playing:
            return
        self._gif_paused = not self._gif_paused
        if self._gif_paused:
            if self._gif_job is not None:
                try:
                    self.after_cancel(self._gif_job)
                except Exception:
                    pass
                self._gif_job = None
        else:
            self._schedule_next_media_frame()
        self._update_gif_button_state()

    def _update_gif_button_state(self):
        if not self._gif_playing:
            self.gif_pause_btn.configure(state="disabled", text="Pauza (p)")
            return
        label = "Wznow (p)" if self._gif_paused else "Pauza (p)"
        self.gif_pause_btn.configure(state="normal", text=label)

    def _save_outputs(self):
        try:
            base = os.path.splitext(os.path.basename(self.img_path))[0]
            suffix = ""
            if self._gif_playing and self._gif_frames:
                suffix = f"_f{self._gif_index + 1:03d}"
            elif self._media_type == "video":
                suffix = f"_f{self._video_frame_index:06d}"

            default_name = f"{base}{suffix}_physics_mask.png"
            initial_dir = os.path.dirname(self.img_path) if self.img_path else None
            out_path = filedialog.asksaveasfilename(
                parent=self,
                title="Zapisz maske",
                defaultextension=".png",
                initialfile=default_name,
                initialdir=initial_dir,
                filetypes=[("PNG", "*.png"), ("Wszystkie pliki", "*.*")],
            )
            if not out_path:
                return

            params = self._read_params()
            mask = detect_shadow_physics(
                self.img_bgr,
                use_geometry_validation=params["use_geometry"],
                use_clahe=params["use_clahe"],
                v_thresh=params["v_thresh"],
                hue_diff_max=params["hue_diff_max"],
                min_area=params["min_area"],
                min_elongation=params["min_elongation"],
                morph_kernel=params["morph_kernel"],
                min_line_conf=params["min_line_conf"],
            )
            cv2.imwrite(out_path, mask)
            self.status_var.set(f"Zapisano: {out_path}")
        except Exception as e:
            self.status_var.set(f"Blad zapisu: {e}")

    def _on_preview_resize(self, event):
        w = int(max(360, getattr(event, "width", 0)))
        h = int(max(260, getattr(event, "height", 0)))
        if w != self.VIEW_W or h != self.VIEW_H:
            self.VIEW_W = w
            self.VIEW_H = h
            self._schedule_render(reason="resize")


def run_app(images_folder: str | None = None):
    app = ShadowTunerApp(images_folder)
    app.mainloop()

