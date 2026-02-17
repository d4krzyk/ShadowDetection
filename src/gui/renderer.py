import threading
import time

import cv2
from PIL import Image, ImageTk

from src.shadow_detection import detect_shadow_physics
from src.shadow_direction import fuse_light_vectors
from src.visualize import build_main_view


class RenderWorkerMixin:
    def _read_params(self):
        mk = self._clamp_int(self.var_morph_kernel.get(), 3, 31)
        if mk % 2 == 0:
            mk += 1
            if mk > 31:
                mk -= 2
        self.var_morph_kernel.set(mk)

        return {
            "v_thresh": self._clamp_int(self.var_v_thresh.get(), 0, 255),
            "hue_diff_max": self._clamp_int(self.var_hue_diff.get(), 0, 90),
            "min_area": self._clamp_int(self.var_min_area.get(), 0, 500000),
            "min_elongation": float(self._clamp_int(self.var_min_elong_x100.get(), 100, 1000)) / 100.0,
            "morph_kernel": mk,
            "min_line_conf": 0.35,
            "use_geometry": bool(self.var_use_geom.get()),
            "use_clahe": bool(self.var_use_clahe.get()),
            "overlay": bool(self.var_overlay.get()),
        }

    def _schedule_render(self, reason: str = "param"):
        with self._render_lock:
            self._render_req_id += 1

        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(target=self._render_worker, daemon=True)
            self._worker_thread.start()

    def _render_worker(self):
        last_done = 0
        while True:
            with self._render_lock:
                req_id = self._render_req_id
            if req_id == last_done:
                return

            last_done = req_id
            params = self._read_params()

            img = self.img_bgr
            if img is None:
                continue

            t0 = time.time()
            mask = detect_shadow_physics(
                img,
                use_geometry_validation=params["use_geometry"],
                use_clahe=params["use_clahe"],
                v_thresh=params["v_thresh"],
                hue_diff_max=params["hue_diff_max"],
                min_area=params["min_area"],
                min_elongation=params["min_elongation"],
                morph_kernel=params["morph_kernel"],
                min_line_conf=params["min_line_conf"],
            )

            vec, conf, ang = None, 0.0, 0.0
            try:
                ang, vec, conf = fuse_light_vectors(img, mask, debug=False)
            except Exception:
                pass

            view = build_main_view(
                img,
                mask,
                overlay=params["overlay"],
                show_direction=True,
                light_vec=vec,
                light_ang=float(ang),
                light_conf=float(conf),
                view_w=self.VIEW_W,
                view_h=self.VIEW_H,
                compass_w=self.COMPASS_W,
            )

            ms = (time.time() - t0) * 1000.0
            self._result_q.put((req_id, view, mask, ms))

    def _poll_results(self):
        latest = None
        try:
            while True:
                latest = self._result_q.get_nowait()
        except Exception:
            pass

        if latest is not None:
            _req_id, view_bgr, _mask, ms = latest
            if view_bgr is not None:
                rgb = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(rgb)
                tk_img = ImageTk.PhotoImage(im)
                self.preview_label.configure(image=tk_img)
                self.preview_label.image = tk_img
                name = self._display_name or ""
                if self.img_path:
                    name = self._display_name or self.img_path
                self.status_var.set(f"{name} | {ms:.0f} ms")

        self.after(50, self._poll_results)

