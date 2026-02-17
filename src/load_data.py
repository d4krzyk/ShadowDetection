import cv2
import os
import random
import numpy as np
from PIL import Image


def load_image(path):
    if path.lower().endswith('.gif'):
        with Image.open(path) as im:
            frame = im.convert('RGB')
            arr = np.array(frame)
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(path)
    if img is None:
        raise ValueError("Nie wczytano obrazu: %s" % path)
    return img

def get_random_image_from_dir(dir_path, include_video: bool = False):
    """Zwraca pełną ścieżkę do losowego pliku obrazu w podanym folderze.
    Obsługiwane rozszerzenia: jpg, jpeg, png, bmp, tiff, gif.
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f"Katalog nie istnieje: {dir_path}")

    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
    if include_video:
        exts = exts + ('.mp4', '.avi', '.mov', '.mkv', '.webm')

    files = [f for f in os.listdir(dir_path)
             if os.path.isfile(os.path.join(dir_path, f))
             and f.lower().endswith(exts)]

    if not files:
        raise ValueError(f"Brak plików obrazu w katalogu: {dir_path}")

    chosen = random.choice(files)
    return os.path.join(dir_path, chosen)
