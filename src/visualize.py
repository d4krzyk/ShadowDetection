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

    if mask_gray.dtype != np.uint8:
        m = mask_gray.astype(np.float32)
        if m.max() <= 1.0:
            m = m * 255.0
        mask_gray = np.clip(m, 0, 255).astype(np.uint8)

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

    if mask_gray.dtype != np.uint8:
        m = mask_gray.astype(np.float32)
        if m.max() <= 1.0:
            m = m * 255.0
        mask_gray = np.clip(m, 0, 255).astype(np.uint8)

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


def _as_gray_mask(mask, target_shape_hw=None):
    if mask is None:
        return None
    if len(mask.shape) == 3:
        m = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        m = mask
    if target_shape_hw is not None and m.shape[:2] != tuple(target_shape_hw):
        m = cv2.resize(m, (int(target_shape_hw[1]), int(target_shape_hw[0])), interpolation=cv2.INTER_NEAREST)
    if m.dtype != np.uint8:
        # najczęstszy przypadek: maska 0/1 lub bool
        m = (m.astype(np.float32))
        if m.max() <= 1.0:
            m = (m * 255.0)
        m = np.clip(m, 0, 255).astype(np.uint8)
    return m


def draw_pca_axis_and_tip(img_bgr, centroid_xy, axis_vec_xy, tip_xy=None, color=(255, 200, 0)):
    """Rysuje oś PCA przechodzącą przez centroid + opcjonalny tip."""
    if img_bgr is None or centroid_xy is None or axis_vec_xy is None:
        return img_bgr

    out = img_bgr
    h, w = out.shape[:2]

    cx, cy = float(centroid_xy[0]), float(centroid_xy[1])
    vx, vy = float(axis_vec_xy[0]), float(axis_vec_xy[1])
    n = (vx * vx + vy * vy) ** 0.5 + 1e-6
    vx /= n
    vy /= n

    L = 0.45 * min(h, w)
    p1 = (int(round(cx - vx * L)), int(round(cy - vy * L)))
    p2 = (int(round(cx + vx * L)), int(round(cy + vy * L)))

    cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)
    cv2.circle(out, (int(round(cx)), int(round(cy))), 4, (255, 255, 255), -1)

    if tip_xy is not None:
        tx, ty = int(round(float(tip_xy[0]))), int(round(float(tip_xy[1])))
        cv2.circle(out, (tx, ty), 5, (0, 255, 255), -1)
        cv2.line(out, (int(round(cx)), int(round(cy))), (tx, ty), (0, 255, 255), 2, cv2.LINE_AA)

    return out


def draw_hough_lines(img_bgr, lines, color=(0, 200, 255)):
    """Rysuje wynik HoughLinesP (Nx1x4) na obrazie."""
    if img_bgr is None or lines is None:
        return img_bgr
    out = img_bgr
    try:
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in l]
            cv2.line(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    except Exception:
        pass
    return out


def overlay_mask_red(img_bgr, mask, alpha=0.35):
    """Nakłada maskę jako czerwony overlay."""
    if img_bgr is None or mask is None:
        return img_bgr
    m = _as_gray_mask(mask, target_shape_hw=img_bgr.shape[:2])
    if m is None:
        return img_bgr
    colored = np.zeros_like(img_bgr)
    colored[:, :, 2] = m
    return cv2.addWeighted(img_bgr, 1.0 - float(alpha), colored, float(alpha), 0)


def draw_vector_arrow(img_bgr, vec_xy, origin=None, length=None, color=(0, 255, 0), thickness=2):
    """Rysuje strzałkę wektora 2D."""
    if img_bgr is None or vec_xy is None:
        return img_bgr
    out = img_bgr
    h, w = out.shape[:2]

    if origin is None:
        origin = (int(0.12 * w), int(0.18 * h))
    ox, oy = int(origin[0]), int(origin[1])

    if length is None:
        length = int(0.18 * min(h, w))

    vx, vy = float(vec_xy[0]), float(vec_xy[1])
    n = (vx * vx + vy * vy) ** 0.5 + 1e-6
    vx /= n
    vy /= n

    ex = int(round(ox + vx * length))
    ey = int(round(oy + vy * length))

    cv2.arrowedLine(out, (ox, oy), (ex, ey), color, int(thickness), tipLength=0.25)
    cv2.circle(out, (ox, oy), 4, color, -1)
    return out


def draw_confidence_badge(img_bgr, confidence, text=None, org=(10, 30)):
    """Mały 'badge' z confidence w rogu."""
    if img_bgr is None:
        return img_bgr
    conf = float(confidence) if confidence is not None else 0.0
    conf = float(np.clip(conf, 0.0, 1.0))

    if text is None:
        text = f"conf={conf:.2f}"

    # kolor od czerwonego do zielonego
    r = int(round(255 * (1.0 - conf)))
    g = int(round(255 * conf))
    col = (0, g, r)

    x, y = int(org[0]), int(org[1])
    pad = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.65
    (tw, th), bl = cv2.getTextSize(text, font, fs, 2)
    cv2.rectangle(img_bgr, (x - pad, y - th - pad), (x + tw + pad, y + bl + pad), (0, 0, 0), -1)
    cv2.rectangle(img_bgr, (x - pad, y - th - pad), (x + tw + pad, y + bl + pad), col, 2)
    cv2.putText(img_bgr, text, (x, y), font, fs, (255, 255, 255), 2, cv2.LINE_AA)
    return img_bgr


def build_light_direction_debug_view(image_bgr, shadow_mask, fuse_dbg, overlay_alpha=0.35):
    """Buduje pojedynczy obraz podglądowy z nałożeniami (do oceny ręcznej).

    fuse_dbg: wynik debug z fuse_light_vectors(..., debug=True)
    """
    if image_bgr is None:
        return None

    out = image_bgr.copy()
    if shadow_mask is not None:
        out = overlay_mask_red(out, shadow_mask, alpha=overlay_alpha)

    # PCA axis + tip
    try:
        m = (fuse_dbg or {}).get('mask')
        if m and m.get('dbg'):
            dbg_m = m['dbg']
            out = draw_pca_axis_and_tip(out, dbg_m.get('centroid_xy'), dbg_m.get('axis_vec'), dbg_m.get('tip_xy'))
    except Exception:
        pass

    # final vector + confidence
    try:
        v_final = (fuse_dbg or {}).get('v_final')
        conf_final = (fuse_dbg or {}).get('conf_final', 0.0)
        if v_final is not None:
            out = draw_vector_arrow(out, v_final, color=(0, 255, 0))
        draw_confidence_badge(out, conf_final, text=f"final conf={float(conf_final):.2f}", org=(10, 30))
    except Exception:
        pass

    # annotate components confidence
    try:
        h = (fuse_dbg or {}).get('hough', {})
        mh = float(h.get('conf', 0.0))
        m = (fuse_dbg or {}).get('mask')
        mm = float(m.get('conf', 0.0)) if m else 0.0
        txt = f"Hough={mh:.2f}  Mask={mm:.2f}"
        cv2.putText(out, txt, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, txt, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    except Exception:
        pass

    return out


def letterbox_to_canvas(img_bgr, canvas_w, canvas_h, bg=(10, 10, 10), inner_margin=12, draw_frame=True):
    """Skaluje obraz do stałej ramki (canvas) z zachowaniem proporcji i paddingiem."""
    if img_bgr is None:
        return None
    ch, cw = img_bgr.shape[:2]
    if ch <= 0 or cw <= 0:
        return None

    canvas_w = int(canvas_w)
    canvas_h = int(canvas_h)
    inner_margin = int(max(0, inner_margin))

    avail_w = max(1, canvas_w - 2 * inner_margin)
    avail_h = max(1, canvas_h - 2 * inner_margin)

    s = min(float(avail_w) / float(cw), float(avail_h) / float(ch))
    s = max(1e-6, s)
    nw = max(1, int(round(cw * s)))
    nh = max(1, int(round(ch * s)))

    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=interp)

    canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
    x0 = (canvas_w - nw) // 2
    y0 = (canvas_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized

    if draw_frame:
        cv2.rectangle(canvas, (0, 0), (canvas_w - 1, canvas_h - 1), (60, 60, 60), 1)
        cv2.rectangle(canvas, (x0, y0), (x0 + nw - 1, y0 + nh - 1), (90, 90, 90), 1)

    return canvas


def render_compass_widget(vec_xy, angle_deg=0.0, confidence=0.0, w=170, h=300, title='LIGHT'):
    """Mały 'kompas' pokazujący kierunek światła strzałką."""
    w = int(max(96, w))
    h = int(max(96, h))
    canvas = np.full((h, w, 3), (14, 14, 14), dtype=np.uint8)

    cx, cy = w // 2, h // 2
    r = int(0.38 * min(w, h))

    if title:
        cv2.putText(canvas, str(title), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)

    cv2.circle(canvas, (cx, cy), r, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 2, (200, 200, 200), -1)

    def put(lbl, px, py):
        cv2.putText(canvas, lbl, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    put('N', cx - 6, cy - r - 10)
    put('S', cx - 6, cy + r + 18)
    put('W', cx - r - 22, cy + 6)
    put('E', cx + r + 10, cy + 6)

    if vec_xy is not None:
        vx, vy = float(vec_xy[0]), float(vec_xy[1])
        n = (vx * vx + vy * vy) ** 0.5 + 1e-6
        vx /= n
        vy /= n
        conf = float(np.clip(float(confidence), 0.0, 1.0))
        length = int(round((r - 6) * (0.35 + 0.65 * conf)))
        ex = int(round(cx + vx * length))
        ey = int(round(cy + vy * length))
        cv2.arrowedLine(canvas, (cx, cy), (ex, ey), (0, 255, 0), 2, tipLength=0.22)

    conf = float(np.clip(float(confidence), 0.0, 1.0))
    ang = float(angle_deg) % 360.0
    cv2.putText(canvas, f"Angle {ang:.0f} deg", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Confidence {conf:.2f}", (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (60, 60, 60), 1)
    return canvas



def build_main_view(image_bgr,
                    shadow_mask,
                    overlay,
                    show_direction,
                    light_vec,
                    light_ang,
                    light_conf,
                    view_w,
                    view_h,
                    compass_w):
    """Składa główny widok: porównanie + kompas.

    Zwraca: view_bgr
    """
    comp = build_comparison(image_bgr, shadow_mask, overlay=overlay)
    if comp is None:
        return None

    comp_fixed = letterbox_to_canvas(comp, view_w, view_h)
    if comp_fixed is None:
        return None

    compass = render_compass_widget(light_vec if show_direction else None,
                                   angle_deg=light_ang if show_direction else 0.0,
                                   confidence=light_conf if show_direction else 0.0,
                                   w=compass_w,
                                   h=view_h)
    if not show_direction:
        cv2.putText(compass, 'DIR OFF', (20, view_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    top_row = cv2.hconcat([comp_fixed, compass])

    return top_row

