import cv2
import os
import random

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Nie wczytano obrazu: %s" % path)
    return img

def get_random_image_from_dir(dir_path):
    """Zwraca pełną ścieżkę do losowego pliku obrazu w podanym folderze.
    Obsługiwane rozszerzenia: jpg, jpeg, png, bmp, tiff.
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f"Katalog nie istnieje: {dir_path}")

    files = [f for f in os.listdir(dir_path)
             if os.path.isfile(os.path.join(dir_path, f))
             and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not files:
        raise ValueError(f"Brak plików obrazu w katalogu: {dir_path}")

    chosen = random.choice(files)
    return os.path.join(dir_path, chosen)
