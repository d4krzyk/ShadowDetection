import os
import random
import cv2

from src.load_data import load_image, get_random_image_from_dir
from src.shadow_detection import detect_shadow_physics
from src.shadow_direction import fuse_light_vectors
from src.visualize import build_light_direction_debug_view


def _list_images(folder):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]


def sample_images(images_folder, n=30, seed=0):
    files = _list_images(images_folder)
    if not files:
        raise ValueError(f"Brak obrazów w: {images_folder}")
    rnd = random.Random(seed)
    rnd.shuffle(files)
    return files[: min(n, len(files))]


def run(images_folder, out_dir='results/light_direction_debug', n=30, seed=0):
    os.makedirs(out_dir, exist_ok=True)

    paths = sample_images(images_folder, n=n, seed=seed)
    print(f"Test: {len(paths)} obrazów z {images_folder}")

    for i, img_path in enumerate(paths, 1):
        img = load_image(img_path)
        mask = detect_shadow_physics(img)

        ang, vec, conf, dbg = fuse_light_vectors(img, mask, debug=True)
        view = build_light_direction_debug_view(img, mask, dbg)

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, f"{i:02d}_{base}_conf_{conf:.2f}_ang_{ang:.1f}.png")
        cv2.imwrite(out_path, view)
        print(f"[{i:02d}/{len(paths)}] conf={conf:.2f} ang={ang:.1f} -> {out_path}")


if __name__ == '__main__':
    # domyślnie train
    folder = os.path.join('data', 'SBU-shadow', 'SBUTrain', 'ShadowImages')
    run(folder, n=30, seed=0)

