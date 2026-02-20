import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.shadow_detection import detect_shadow_physics
from src.shadow_direction import (
    estimate_light_direction_bgr,
    estimate_light_direction_shadow_edges,
    estimate_light_direction_shadow_edge_gradient,
    estimate_from_mask_pca_tip,
    is_object_visible_opposite_shadow,
    merge_vectors,
    _normalize2,
)
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
            "use_soft_shadow": bool(self.var_soft_shadow.get()),
            "soft_shadow_thresh": float(self._clamp_int(self.var_soft_shadow_thresh_x100.get(), 5, 90)) / 100.0,
            "soft_shadow_scale": float(self._clamp_int(self.var_soft_shadow_scale_x100.get(), 10, 120)) / 100.0,
            "show_soft_mask": bool(self.var_show_soft_mask.get()),
            "use_dir_hough": bool(self.var_use_dir_hough.get()),
            "use_dir_pca": bool(self.var_use_dir_pca.get()),
            "use_dir_grad": bool(self.var_use_dir_grad.get()),
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
            params_key = tuple(sorted(params.items()))
            if getattr(self, "_mean_conf_params_key", None) != params_key:
                self._mean_conf_params_key = params_key
                self._mean_conf_count = 0
                self._mean_conf_running = 0.0

            img = self.img_bgr
            if img is None:
                continue

            t0 = time.time()
            if params["use_soft_shadow"]:
                mask, dbg = detect_shadow_physics(
                    img,
                    use_geometry_validation=params["use_geometry"],
                    use_clahe=params["use_clahe"],
                    v_thresh=params["v_thresh"],
                    hue_diff_max=params["hue_diff_max"],
                    min_area=params["min_area"],
                    min_elongation=params["min_elongation"],
                    morph_kernel=params["morph_kernel"],
                    min_line_conf=params["min_line_conf"],
                    use_soft_shadow=params["use_soft_shadow"],
                    soft_shadow_thresh=params["soft_shadow_thresh"],
                    soft_shadow_scale=params["soft_shadow_scale"],
                    debug=True,
                )
                if isinstance(dbg, dict):
                    overlay_mask = dbg.get("dir_weight_map")
                    if overlay_mask is None:
                        overlay_mask = dbg.get("shadow_score")
                else:
                    overlay_mask = None
            else:
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
                    use_soft_shadow=params["use_soft_shadow"],
                    soft_shadow_thresh=params["soft_shadow_thresh"],
                    soft_shadow_scale=params["soft_shadow_scale"],
                )
                overlay_mask = None

            use_soft = bool(params.get("show_soft_mask", True))
            display_mask = overlay_mask if (use_soft and overlay_mask is not None) else mask

            vec, conf, ang = None, None, None
            mean_conf = None
            mask_valid = mask is not None and int(np.count_nonzero(mask)) >= 50
            if mask_valid:
                try:
                    use_h = bool(params.get("use_dir_hough", True))
                    use_p = bool(params.get("use_dir_pca", True))
                    use_g = bool(params.get("use_dir_grad", True))
                    if not (use_h or use_p or use_g):
                        use_h = use_p = use_g = True

                    components = []
                    conf_h = None
                    vec_h = None
                    vec_g = None
                    v_m = None

                    if use_h:
                        ang_h, vec_h, conf_h = estimate_light_direction_shadow_edges(img)

                    if use_g:
                        ang_g, vec_g, conf_g = estimate_light_direction_shadow_edge_gradient(img, mask, weight_map=overlay_mask)
                        components.append({"name": "grad_edge", "vec": vec_g, "conf": float(max(0.10, conf_g))})

                    if use_p:
                        pca_out = estimate_from_mask_pca_tip(mask, weight_map=overlay_mask, debug=True)
                        v_m, conf_m, _dbg_m = pca_out
                        visible, conf_obj, _dbg_obj = is_object_visible_opposite_shadow(
                            img,
                            mask,
                            _normalize2(-np.array(v_m, dtype=np.float32)),
                            debug=True,
                        )
                        conf_m2 = float(np.clip(conf_m + 0.25 * conf_obj, 0.0, 1.0))
                        components.append({"name": "mask", "vec": v_m, "conf": conf_m2})

                    if use_h and vec_h is not None and conf_h is not None:
                        if vec_g is not None and v_m is not None:
                            vh = _normalize2(vec_h)
                            vg = _normalize2(vec_g)
                            vm = _normalize2(v_m)
                            if float(np.dot(vm, vg)) > 0.0 and float(np.dot(vh, vm)) < 0.0 and float(np.dot(vh, vg)) < 0.0:
                                conf_h = float(conf_h) * 0.35
                        components.append({"name": "hough", "vec": vec_h, "conf": float(conf_h)})

                    v_final, conf = merge_vectors(components)
                    if conf is not None and float(conf) > 1e-6:
                        ang = (np.degrees(np.arctan2(float(v_final[1]), float(v_final[0]))) + 360.0) % 360.0
                        vec = (float(v_final[0]), float(v_final[1]))
                        mean_conf = float(np.clip(float(conf), 0.0, 1.0))
                        self._mean_conf_count = int(getattr(self, "_mean_conf_count", 0)) + 1
                        prev = float(getattr(self, "_mean_conf_running", 0.0))
                        self._mean_conf_running = prev + (mean_conf - prev) / float(self._mean_conf_count)
                    else:
                        vec, conf, ang = None, None, None
                except Exception:
                    vec, conf, ang = None, None, None
                    mean_conf = None

            mean_conf_display = float(getattr(self, "_mean_conf_running", 0.0)) if getattr(self, "_mean_conf_count", 0) > 0 else None

            view = build_main_view(
                img,
                display_mask,
                overlay=params["overlay"],
                show_direction=True,
                light_vec=vec,
                light_ang=ang,
                light_conf=conf,
                mean_conf=mean_conf_display,
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
                target_w = int(self.preview_label.winfo_width() or 0)
                target_h = int(self.preview_label.winfo_height() or 0)
                if target_w > 10 and target_h > 10:
                    view_bgr = cv2.resize(view_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
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
