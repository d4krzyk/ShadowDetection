import cv2
import numpy as np

def show(title, img, max_size=(1200, 800)):
    """Proste pokazanie obrazu w resizowalnym oknie (przydatne w PyCharm).

    Parametry:
    - title: tytuł okna
    - img: obraz
    - max_size: maksymalny rozmiar okna (szerokość, wysokość)
    """
    if img is None:
        raise ValueError("Pusty obraz")

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    h, w = img.shape[:2]
    cap_w, cap_h = max_size
    win_w = min(w, cap_w)
    win_h = min(h, cap_h)
    try:
        cv2.resizeWindow(title, win_w, win_h)
    except Exception:
        # nie wszystkie backendy obsługują resizeWindow; to nie jest krytyczne
        pass

    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_comparison(title, image, mask, overlay=True, scale=1.0, max_size=(1200, 800)):
    """Wyświetla porównanie oryginalnego obrazu i maski cieni w jednym oknie.

    Parametry:
    - title: tytuł okna
    - image: obraz BGR (numpy.ndarray)
    - mask: jednokanałowa maska (0..255) lub obraz binarny
    - overlay: jeśli True, po prawej pokaże obraz z kolorową nakładką maski; jeśli False, pokaże samą kolorowaną maskę
    - scale: współczynnik skalowania (np. 0.5 zmniejszy do 50%)
    - max_size: maksymalny rozmiar okna (szerokość, wysokość)
    """
    if image is None:
        raise ValueError("Pusty obraz")
    if mask is None:
        raise ValueError("Pusta maska")

    # Upewnij się, że maska jest jednokanałowa i ma zakres 0..255
    if len(mask.shape) == 3:
        # jeśli maska ma 3 kanały, skonwertuj do szarości
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask

    # Normalizacja maski do 0..255 dtype=uint8
    mask_gray = cv2.normalize(mask_gray, None, 0, 255, cv2.NORM_MINMAX)
    if mask_gray.dtype != np.uint8:
        mask_gray = mask_gray.astype(np.uint8)

    # Koloryzacja maski (czerwony kanał)
    mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    colored_mask = np.zeros_like(mask_bgr)
    # Ustaw czerwony kanał na wartości maski -> czerwone zaznaczenia
    colored_mask[:, :, 2] = mask_gray

    # Przygotuj wersję do prawej kolumny: overlay albo sama kolorowa maska
    if overlay:
        # Dopasuj typ i rozmiar
        base = image.copy()
        if base.shape != colored_mask.shape:
            # zmień rozmiar kolorowanej maski do rozmiaru obrazu
            colored_mask = cv2.resize(colored_mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay_img = cv2.addWeighted(base, 0.7, colored_mask, 0.3, 0)
        right = overlay_img
    else:
        # Pokaż samą kolorowaną maskę dopasowaną do rozmiaru obrazu
        if image.shape != mask_bgr.shape:
            mask_bgr = cv2.resize(mask_bgr, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        right = mask_bgr

    left = image

    # Skalowanie, jeśli wymagane
    if scale <= 0 or scale > 1.5:
        scale = 1.0
    if scale != 1.0:
        new_w = int(left.shape[1] * scale)
        new_h = int(left.shape[0] * scale)
        left = cv2.resize(left, (new_w, new_h), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Sklejenie boczne
    try:
        comparison = cv2.hconcat([left, right])
    except Exception:
        h = min(left.shape[0], right.shape[0])
        left_r = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
        right_r = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
        comparison = cv2.hconcat([left_r, right_r])

    # Pokaz w resizowalnym oknie - przydatne np. w PyCharm
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    ch, cw = comparison.shape[:2]
    cap_w, cap_h = max_size
    win_w = min(cw, cap_w)
    win_h = min(ch, cap_h)
    try:
        cv2.resizeWindow(title, win_w, win_h)
    except Exception:
        pass

    cv2.imshow(title, comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def build_comparison(image, mask, overlay=True):
    """Zwraca obraz porównania (left=image, right=overlay mask) bez wywoływania waitKey.
    """
    if image is None or mask is None:
        return None

    # ensure single-channel mask
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask

    mask_gray = cv2.normalize(mask_gray, None, 0, 255, cv2.NORM_MINMAX)
    if mask_gray.dtype != np.uint8:
        mask_gray = mask_gray.astype(np.uint8)

    mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    colored_mask = np.zeros_like(mask_bgr)
    colored_mask[:, :, 2] = mask_gray

    if overlay:
        base = image.copy()
        if base.shape != colored_mask.shape:
            colored_mask = cv2.resize(colored_mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
        right = cv2.addWeighted(base, 0.7, colored_mask, 0.3, 0)
    else:
        if image.shape != mask_bgr.shape:
            mask_bgr = cv2.resize(mask_bgr, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        right = mask_bgr

    left = image
    if left.shape[0] != right.shape[0]:
        h = min(left.shape[0], right.shape[0])
        left = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
        right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))

    comp = cv2.hconcat([left, right])
    return comp


def draw_text_box(img, text, org=(10, 30), font_scale=0.6, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Rysuje tekst z tłem dla czytelności."""
    x, y = int(org[0]), int(org[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
    pad = 4
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, 1, cv2.LINE_AA)


def draw_light_vector(img_bgr, angle_deg, vec_xy=None, origin=None, length=None, color=(0, 255, 255)):
    """Rysuje strzałkę pokazującą kierunek światła w 2D.

    Uwaga: z pojedynczego obrazu nie da się wiarygodnie wyznaczyć pełnego wektora 3D (bez geometrii/sceny).
    To jest *kierunek na płaszczyźnie obrazu* (2D), który zwykle koreluje z kierunkiem oświetlenia/cieni.

    - angle_deg: kąt 0..360 (0=→, 90=↓)
    - vec_xy: opcjonalny (vx, vy); jeśli podasz, angle_deg użyty tylko do opisu
    """
    if img_bgr is None:
        return img_bgr

    h, w = img_bgr.shape[:2]
    out = img_bgr

    if origin is None:
        origin = (int(0.12 * w), int(0.18 * h))
    ox, oy = int(origin[0]), int(origin[1])

    if length is None:
        length = int(0.18 * min(h, w))

    if vec_xy is None:
        ang = np.deg2rad(float(angle_deg))
        vx = float(np.cos(ang))
        vy = float(np.sin(ang))
    else:
        vx, vy = float(vec_xy[0]), float(vec_xy[1])
        norm = (vx * vx + vy * vy) ** 0.5 + 1e-6
        vx /= norm
        vy /= norm

    ex = int(ox + vx * length)
    ey = int(oy + vy * length)

    # arrow
    cv2.arrowedLine(out, (ox, oy), (ex, ey), color, 2, tipLength=0.25)
    # origin dot
    cv2.circle(out, (ox, oy), 4, color, -1)

    draw_text_box(out, f"Light dir: {angle_deg:.1f} deg", org=(ox, max(25, oy - 10)), font_scale=0.55, color=(255, 255, 255), bg_color=(0, 0, 0))
    return out
