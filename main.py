from src.load_data import load_image, get_random_image_from_dir
from src.shadow_detection import detect_shadow_physics
from src.visualize import build_comparison
from src.visualize import draw_text_box

import os
import cv2


if __name__ == '__main__':
    images_folder = "data/SBU-shadow/SBUTrain/ShadowImages"

    win = 'Shadow Detection (Physics)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    use_geometry = True
    use_clahe = True

    # --- Parametry do strojenia (najważniejsze) ---
    v_thresh = 75
    hue_diff_max = 10
    min_area = 250
    min_elongation = 1.30
    morph_kernel = 5
    min_line_conf = 0.35

    show_help = True

    def run_detector(img_bgr):
        return detect_shadow_physics(
            img_bgr,
            use_geometry_validation=use_geometry,
            use_clahe=use_clahe,
            v_thresh=v_thresh,
            hue_diff_max=hue_diff_max,
            min_area=min_area,
            min_elongation=min_elongation,
            morph_kernel=morph_kernel,
            min_line_conf=min_line_conf,
        )

    img_path = get_random_image_from_dir(images_folder)
    img = load_image(img_path)

    print("=== Shadow Detection - podejście fizyczne ===")
    print("Instrukcja: r - losuj nowy obraz, g - geometria ON/OFF, c - CLAHE ON/OFF, h - help ON/OFF,")
    print("            s - zapisz maskę, q/ESC - wyjście")
    print("Tuning:")
    print("  v/V: v_thresh -/+5 | u/U: hue_diff -/+2 | a/A: min_area -/+50")
    print("  e/E: elongation -/+0.05 | k/K: morph_kernel -/+2 | l/L: min_line_conf -/+0.05")
    print(f"Geometria: {'ON' if use_geometry else 'OFF'} | CLAHE: {'ON' if use_clahe else 'OFF'} | Plik: {img_path}")

    mask = run_detector(img)

    while True:
        # buduj porównanie
        comp = build_comparison(img, mask, overlay=True)
        if comp is None:
            print("Błąd: brak obrazu lub maski")
            break

        # overlay parametrów na lewym obrazie (czyli na jego kopii) – najprościej: rysuj na comp.
        # comp ma układ [left|right], więc tekst rysujemy w obrębie lewego panelu.
        if show_help:
            draw_text_box(comp, f"v_thresh={v_thresh}  hue_diff={hue_diff_max}  area>={min_area}", org=(10, 25), font_scale=0.55)
            draw_text_box(comp, f"elong>={min_elongation:.2f}  morph={morph_kernel}  line_conf>={min_line_conf:.2f}", org=(10, 50), font_scale=0.55)
            draw_text_box(comp, f"g:Geom({'ON' if use_geometry else 'OFF'})  c:CLAHE({'ON' if use_clahe else 'OFF'})  r:Rand  s:Save  h:Help", org=(10, 75), font_scale=0.55)

        h, w = comp.shape[:2]
        max_w, max_h = 1200, 800
        win_w = min(w, max_w)
        win_h = min(h, max_h)
        try:
            cv2.resizeWindow(win, win_w, win_h)
        except Exception:
            pass

        cv2.imshow(win, comp)
        key = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('g'):
            use_geometry = not use_geometry
            print(f"Geometria: {'ON' if use_geometry else 'OFF'} | CLAHE: {'ON' if use_clahe else 'OFF'}")
            mask = run_detector(img)
        elif key == ord('c'):
            use_clahe = not use_clahe
            print(f"Geometria: {'ON' if use_geometry else 'OFF'} | CLAHE: {'ON' if use_clahe else 'OFF'}")
            mask = run_detector(img)

        # --- tuning keys ---
        elif key == ord('v'):
            v_thresh = max(0, v_thresh - 5)
            mask = run_detector(img)
        elif key == ord('V'):
            v_thresh = min(255, v_thresh + 5)
            mask = run_detector(img)
        elif key == ord('u'):
            hue_diff_max = max(0, hue_diff_max - 2)
            mask = run_detector(img)
        elif key == ord('U'):
            hue_diff_max = min(90, hue_diff_max + 2)
            mask = run_detector(img)
        elif key == ord('a'):
            min_area = max(0, min_area - 50)
            mask = run_detector(img)
        elif key == ord('A'):
            min_area = min(50000, min_area + 50)
            mask = run_detector(img)
        elif key == ord('e'):
            min_elongation = max(1.0, round(min_elongation - 0.05, 2))
            mask = run_detector(img)
        elif key == ord('E'):
            min_elongation = min(10.0, round(min_elongation + 0.05, 2))
            mask = run_detector(img)
        elif key == ord('k'):
            morph_kernel = max(3, morph_kernel - 2)
            if morph_kernel % 2 == 0:
                morph_kernel -= 1
            mask = run_detector(img)
        elif key == ord('K'):
            morph_kernel = min(51, morph_kernel + 2)
            if morph_kernel % 2 == 0:
                morph_kernel += 1
            mask = run_detector(img)
        elif key == ord('l'):
            min_line_conf = max(0.0, round(min_line_conf - 0.05, 2))
            mask = run_detector(img)
        elif key == ord('L'):
            min_line_conf = min(1.0, round(min_line_conf + 0.05, 2))
            mask = run_detector(img)

        elif key == ord('r'):
            try:
                img_path = get_random_image_from_dir(images_folder)
                img = load_image(img_path)
                print(f"\n=== Nowy obraz: {img_path} | Geometria: {'ON' if use_geometry else 'OFF'} | CLAHE: {'ON' if use_clahe else 'OFF'} ===")
                mask = run_detector(img)
            except Exception as e:
                print(f"Błąd przy ładowaniu nowego obrazu: {e}")
        elif key == ord('s'):
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_mask = f"{base}_physics_mask.png"
            out_overlay = f"{base}_physics_overlay.png"
            cv2.imwrite(out_mask, mask)
            overlay_img = build_comparison(img, mask, overlay=True)
            cv2.imwrite(out_overlay, overlay_img)
            print(f"Zapisano: {out_mask}, {out_overlay}")

    cv2.destroyAllWindows()
