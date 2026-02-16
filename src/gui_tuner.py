import os
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from src.load_data import load_image, get_random_image_from_dir
from src.shadow_detection import detect_shadow_physics
from src.shadow_direction import fuse_light_vectors
from src.visualize import build_main_view


def _clamp_int(v, lo, hi):
    return int(max(lo, min(hi, int(v))))


class ShadowTunerApp(tk.Tk):
    """GUI do strojenia detekcji cieni (Tkinter).

    - suwaki: parametry liczbowe
    - checkboxy/przyciski: GEOM/CLAHE/overlay/dir/panel
    - render w tle (wątek), żeby UI się nie zawieszało
    """

    def __init__(self, images_folder: str):
        super().__init__()

        self.title('ShadowDetection - Tuner (Tkinter)')
        self.minsize(1100, 520)

        self.images_folder = images_folder
        self.img_path = get_random_image_from_dir(images_folder)
        self.img_bgr = load_image(self.img_path)

        # stały rozmiar podglądu
        self.VIEW_W = 800
        self.VIEW_H = 300
        self.PANEL_H = 85
        self.COMPASS_W = 170

        # --- model stanu ---
        self.var_v_thresh = tk.IntVar(value=75)
        self.var_hue_diff = tk.IntVar(value=10)
        self.var_min_area = tk.IntVar(value=250)
        self.var_min_elong_x100 = tk.IntVar(value=130)
        self.var_morph_kernel = tk.IntVar(value=5)
        self.var_line_conf_x100 = tk.IntVar(value=35)

        self.var_use_geom = tk.BooleanVar(value=True)
        self.var_use_clahe = tk.BooleanVar(value=True)
        self.var_overlay = tk.BooleanVar(value=True)
        self.var_show_dir = tk.BooleanVar(value=True)
        self.var_show_panel = tk.BooleanVar(value=True)

        self._render_lock = threading.Lock()
        self._render_req_id = 0
        self._worker_thread = None
        self._result_q: queue.Queue = queue.Queue()

        self._build_ui()
        self._bind_events()

        self.status_var.set(f'Start: {self.img_path}')
        self._schedule_render(reason='init')
        self.after(50, self._poll_results)

    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky='nsw')

        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky='nsew')
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        # --- kontrolki ---
        ttk.Label(left, text='Parametry', font=('Segoe UI', 11, 'bold')).grid(row=0, column=0, sticky='w')

        r = 1
        r = self._add_scale(left, r, 'v_thresh (0-255)', self.var_v_thresh, 0, 255)
        r = self._add_scale(left, r, 'hue_diff (0-40)', self.var_hue_diff, 0, 40)
        r = self._add_scale(left, r, 'min_area (0-5000)', self.var_min_area, 0, 5000)
        r = self._add_scale(left, r, 'min_elong*100 (100-400)', self.var_min_elong_x100, 100, 400)
        r = self._add_scale(left, r, 'morph_kernel (3-31 odd)', self.var_morph_kernel, 3, 31)
        r = self._add_scale(left, r, 'line_conf*100 (0-100)', self.var_line_conf_x100, 0, 100)

        ttk.Separator(left).grid(row=r, column=0, sticky='ew', pady=8)
        r += 1

        ttk.Label(left, text='Opcje', font=('Segoe UI', 11, 'bold')).grid(row=r, column=0, sticky='w')
        r += 1

        ttk.Checkbutton(left, text='GEOM (walidacja geometryczna)', variable=self.var_use_geom, command=self._schedule_render).grid(row=r, column=0, sticky='w')
        r += 1
        ttk.Checkbutton(left, text='CLAHE (lokalny kontrast)', variable=self.var_use_clahe, command=self._schedule_render).grid(row=r, column=0, sticky='w')
        r += 1
        ttk.Checkbutton(left, text='Overlay (maska jako nakładka)', variable=self.var_overlay, command=self._schedule_render).grid(row=r, column=0, sticky='w')
        r += 1
        ttk.Checkbutton(left, text='Kierunek światła (kompas)', variable=self.var_show_dir, command=self._schedule_render).grid(row=r, column=0, sticky='w')
        r += 1
        ttk.Checkbutton(left, text='Panel statusu', variable=self.var_show_panel, command=self._schedule_render).grid(row=r, column=0, sticky='w')
        r += 1

        ttk.Separator(left).grid(row=r, column=0, sticky='ew', pady=8)
        r += 1

        btns = ttk.Frame(left)
        btns.grid(row=r, column=0, sticky='ew')
        ttk.Button(btns, text='Losuj obraz (r)', command=self._random_image).grid(row=0, column=0, sticky='ew')
        ttk.Button(btns, text='Zapisz (s)', command=self._save_outputs).grid(row=0, column=1, sticky='ew', padx=(8, 0))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        r += 1

        self.status_var = tk.StringVar(value='')
        ttk.Label(left, textvariable=self.status_var, wraplength=320, foreground='#444').grid(row=r, column=0, sticky='w', pady=(10, 0))

        # --- podgląd ---
        self.preview_label = ttk.Label(right)
        self.preview_label.grid(row=0, column=0, sticky='nsew')

    def _add_scale(self, parent, row, label, var, frm, to):
        box = ttk.Frame(parent)
        box.grid(row=row, column=0, sticky='ew', pady=2)
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text=label).grid(row=0, column=0, sticky='w')
        scale = ttk.Scale(box, from_=frm, to=to, orient='horizontal', command=lambda _v: self._on_scale(var))
        scale.grid(row=0, column=1, sticky='ew', padx=(8, 8))

        # ustaw wartość startową
        scale.set(var.get())

        value_lbl = ttk.Label(box, width=5, anchor='e')
        value_lbl.grid(row=0, column=2, sticky='e')
        value_lbl.configure(text=str(var.get()))

        def update_label(*_):
            value_lbl.configure(text=str(int(var.get())))

        var.trace_add('write', update_label)
        return row + 1

    def _on_scale(self, var):
        # ttk.Scale daje float, więc zaokrąglamy
        var.set(int(float(var.get())))
        self._schedule_render()

    def _bind_events(self):
        self.bind('<KeyPress-r>', lambda _e: self._random_image())
        self.bind('<KeyPress-s>', lambda _e: self._save_outputs())
        self.bind('<Escape>', lambda _e: self.destroy())
        self.bind('<KeyPress-q>', lambda _e: self.destroy())

    def _random_image(self):
        try:
            self.img_path = get_random_image_from_dir(self.images_folder)
            self.img_bgr = load_image(self.img_path)
            self.status_var.set(f'Nowy obraz: {self.img_path}')
            self._schedule_render(reason='random')
        except Exception as e:
            self.status_var.set(f'Błąd losowania: {e}')

    def _read_params(self):
        mk = _clamp_int(self.var_morph_kernel.get(), 3, 31)
        if mk % 2 == 0:
            mk += 1
            if mk > 31:
                mk -= 2
        self.var_morph_kernel.set(mk)

        return {
            'v_thresh': _clamp_int(self.var_v_thresh.get(), 0, 255),
            'hue_diff_max': _clamp_int(self.var_hue_diff.get(), 0, 90),
            'min_area': _clamp_int(self.var_min_area.get(), 0, 500000),
            'min_elongation': float(_clamp_int(self.var_min_elong_x100.get(), 100, 1000)) / 100.0,
            'morph_kernel': mk,
            'min_line_conf': float(_clamp_int(self.var_line_conf_x100.get(), 0, 100)) / 100.0,
            'use_geometry': bool(self.var_use_geom.get()),
            'use_clahe': bool(self.var_use_clahe.get()),
            'overlay': bool(self.var_overlay.get()),
            'show_dir': bool(self.var_show_dir.get()),
            'show_panel': bool(self.var_show_panel.get()),
        }

    def _schedule_render(self, reason: str = 'param'):
        # debounce: jeśli worker działa, podbij request id
        with self._render_lock:
            self._render_req_id += 1
            req_id = self._render_req_id

        # uruchom worker jeśli nie działa
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(target=self._render_worker, daemon=True)
            self._worker_thread.start()

    def _render_worker(self):
        # pętla: pobierz najnowszy req_id i wyrenderuj
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
                use_geometry_validation=params['use_geometry'],
                use_clahe=params['use_clahe'],
                v_thresh=params['v_thresh'],
                hue_diff_max=params['hue_diff_max'],
                min_area=params['min_area'],
                min_elongation=params['min_elongation'],
                morph_kernel=params['morph_kernel'],
                min_line_conf=params['min_line_conf'],
            )

            vec, conf, ang = None, 0.0, 0.0
            try:
                ang, vec, conf = fuse_light_vectors(img, mask, debug=False)
            except Exception:
                pass

            params_lines = [
                f"v={params['v_thresh']} hue={params['hue_diff_max']} area={params['min_area']} elong>={params['min_elongation']:.2f}",
                f"morph={params['morph_kernel']} line_conf>={params['min_line_conf']:.2f} | GEOM={'ON' if params['use_geometry'] else 'OFF'} CLAHE={'ON' if params['use_clahe'] else 'OFF'} Overlay={'ON' if params['overlay'] else 'OFF'}",
                f"Light: ang={float(ang):.0f}deg conf={float(conf):.2f}" if params['show_dir'] else "Light: -",
            ]
            help_line = "r:losuj  s:zapisz  q/ESC:wyjście"

            view = build_main_view(
                img,
                mask,
                overlay=params['overlay'],
                show_direction=params['show_dir'],
                light_vec=vec,
                light_ang=float(ang),
                light_conf=float(conf),
                view_w=self.VIEW_W,
                view_h=self.VIEW_H,
                compass_w=self.COMPASS_W,
                panel_h=(self.PANEL_H if params['show_panel'] else 0),
                params_lines=params_lines,
                help_line=help_line,
                buttons=None,
                button_state=None,
            )

            ms = (time.time() - t0) * 1000.0
            self._result_q.put((req_id, view, mask, float(ang), float(conf), ms))

    def _poll_results(self):
        # bierzemy tylko najnowszy wynik
        latest = None
        try:
            while True:
                latest = self._result_q.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            req_id, view_bgr, mask, ang, conf, ms = latest
            if view_bgr is not None:
                rgb = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(rgb)
                tk_img = ImageTk.PhotoImage(im)
                self.preview_label.configure(image=tk_img)
                self.preview_label.image = tk_img
                self.status_var.set(f"{os.path.basename(self.img_path)} | ang={ang:.0f} conf={conf:.2f} | {ms:.0f} ms")

        self.after(50, self._poll_results)

    def _save_outputs(self):
        try:
            base = os.path.splitext(os.path.basename(self.img_path))[0]
            params = self._read_params()
            mask = detect_shadow_physics(
                self.img_bgr,
                use_geometry_validation=params['use_geometry'],
                use_clahe=params['use_clahe'],
                v_thresh=params['v_thresh'],
                hue_diff_max=params['hue_diff_max'],
                min_area=params['min_area'],
                min_elongation=params['min_elongation'],
                morph_kernel=params['morph_kernel'],
                min_line_conf=params['min_line_conf'],
            )
            out_mask = f"{base}_physics_mask.png"
            cv2.imwrite(out_mask, mask)
            self.status_var.set(f"Zapisano: {out_mask}")
        except Exception as e:
            self.status_var.set(f"Błąd zapisu: {e}")


def run_app(images_folder: str):
    app = ShadowTunerApp(images_folder)
    app.mainloop()


if __name__ == '__main__':
    folder = os.path.join('data', 'SBU-shadow', 'SBUTrain', 'ShadowImages')
    run_app(folder)
