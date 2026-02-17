import os

import cv2
import numpy as np
from PIL import Image, ImageSequence

from src.load_data import load_image


class MediaPlayerMixin:
    def _is_gif(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() == ".gif"

    def _is_video(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm")

    def _load_gif(self, path: str):
        frames = []
        durations = []
        with Image.open(path) as im:
            for frame in ImageSequence.Iterator(im):
                rgb = frame.convert("RGB")
                frames.append(rgb.copy())
                dur = int(frame.info.get("duration", im.info.get("duration", 100)) or 100)
                if dur <= 0:
                    dur = 100
                durations.append(dur)
        return frames, durations

    def _set_current_gif_frame(self, index: int):
        frame = self._gif_frames[index]
        arr = np.array(frame)
        self.img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        base = os.path.basename(self.img_path)
        self._display_name = f"{base} [gif {index + 1}/{len(self._gif_frames)}]"

    def _schedule_next_gif_frame(self):
        if not self._gif_playing or self._gif_paused:
            return
        dur = self._gif_durations[self._gif_index] if self._gif_durations else 100
        dur = int(max(20, min(1000, dur)))
        if self._gif_job is not None:
            try:
                self.after_cancel(self._gif_job)
            except Exception:
                pass
        self._gif_job = self.after(dur, self._advance_gif_frame)

    def _schedule_next_video_frame(self):
        if not self._gif_playing or self._gif_paused:
            return
        fps = float(self._video_fps or 0.0)
        if fps <= 1.0:
            fps = 25.0
        delay_ms = int(max(20, min(1000, 1000.0 / fps)))
        if self._gif_job is not None:
            try:
                self.after_cancel(self._gif_job)
            except Exception:
                pass
        self._gif_job = self.after(delay_ms, self._advance_video_frame)

    def _schedule_next_media_frame(self):
        if self._media_type == "video":
            self._schedule_next_video_frame()
        elif self._media_type == "gif":
            self._schedule_next_gif_frame()

    def _advance_gif_frame(self):
        if not self._gif_playing or not self._gif_frames or self._gif_paused:
            return
        self._gif_index = (self._gif_index + 1) % len(self._gif_frames)
        self._set_current_gif_frame(self._gif_index)
        self._schedule_render(reason="gif")
        self._schedule_next_gif_frame()

    def _advance_video_frame(self):
        if not self._gif_playing or self._gif_paused or self._video_cap is None:
            return
        ok, frame = self._video_cap.read()
        if not ok or frame is None:
            # loop to start
            self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._video_frame_index = 0
            ok, frame = self._video_cap.read()
            if not ok or frame is None:
                return
        self._video_frame_index += 1
        self.img_bgr = frame
        base = os.path.basename(self.img_path)
        self._display_name = f"{base} [vid {self._video_frame_index}]"
        self._schedule_render(reason="video")
        self._schedule_next_video_frame()

    def _set_image_source(self, path: str):
        self.img_path = path
        if self._gif_job is not None:
            try:
                self.after_cancel(self._gif_job)
            except Exception:
                pass
            self._gif_job = None

        if self._video_cap is not None:
            try:
                self._video_cap.release()
            except Exception:
                pass
            self._video_cap = None

        if self._is_gif(path):
            frames, durations = self._load_gif(path)
            if not frames:
                raise ValueError(f"Brak klatek w GIF: {path}")
            self._gif_frames = frames
            self._gif_durations = durations
            self._gif_index = 0
            self._gif_playing = True
            self._gif_paused = False
            self._media_type = "gif"
            self._set_current_gif_frame(0)
            self._update_gif_button_state()
            self._schedule_render(reason="gif-init")
            self._schedule_next_gif_frame()
        elif self._is_video(path):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError(f"Nie mozna otworzyc wideo: {path}")
            self._video_cap = cap
            self._video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            self._video_frame_index = 0
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                self._video_cap = None
                raise ValueError(f"Brak klatek w wideo: {path}")
            self.img_bgr = frame
            self._gif_frames = []
            self._gif_durations = []
            self._gif_index = 0
            self._gif_playing = True
            self._gif_paused = False
            self._media_type = "video"
            self._display_name = f"{os.path.basename(path)} [vid 1]"
            self._update_gif_button_state()
            self._schedule_render(reason="video-init")
            self._schedule_next_video_frame()
        else:
            self._gif_frames = []
            self._gif_durations = []
            self._gif_index = 0
            self._gif_playing = False
            self._gif_paused = False
            self._media_type = None
            self._update_gif_button_state()
            self.img_bgr = load_image(path)
            self._display_name = os.path.basename(path)

