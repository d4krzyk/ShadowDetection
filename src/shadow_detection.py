import cv2
import numpy as np


def _circular_hue_diff(h1, h2):
    """Różnica H w HSV (OpenCV: H 0..179) z uwzględnieniem cykliczności."""
    d = np.abs(h1.astype(np.int16) - h2.astype(np.int16))
    d = np.minimum(d, 180 - d)
    return d.astype(np.uint8)


def _dominant_line_angle_deg(edge_img, min_line_length=40, max_line_gap=10, hough_thresh=60):
    """Zwraca (angle_deg, confidence) z HoughLinesP.

    angle_deg w zakresie [0,180) opisuje orientację linii (nie wektora).
    confidence 0..1 mówi jak spójne są kąty (ważone długością).
    """
    lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180.0, threshold=int(hough_thresh),
                            minLineLength=int(min_line_length), maxLineGap=int(max_line_gap))
    if lines is None or len(lines) == 0:
        return None, 0.0

    angs = []
    wts = []
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in l]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1.0:
            continue
        theta = np.arctan2(dy, dx)  # -pi..pi
        theta = float(theta % np.pi)
        angs.append(theta)
        wts.append(length)

    if not angs:
        return None, 0.0

    angs = np.array(angs, dtype=np.float32)
    wts = np.array(wts, dtype=np.float32)

    c = float(np.sum(np.cos(2 * angs) * wts))
    s = float(np.sum(np.sin(2 * angs) * wts))
    mean = 0.5 * np.arctan2(s, c)
    mean = float(mean % np.pi)

    R = ((c * c + s * s) ** 0.5) / (float(np.sum(wts)) + 1e-6)
    conf = float(np.clip(R, 0.0, 1.0))

    return float(np.degrees(mean)), conf


def detect_shadow_physics(
        image_bgr,
        v_thresh=75,
        hue_window=31,
        hue_diff_max=10,
        min_area=250,
        min_elongation=1.3,
        morph_kernel=5,
        # local contrast (luminance) enhancement
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        # geometry validation (binary filter)
        use_geometry_validation=True,
        min_line_conf=0.35,
        debug=False,
):
    """Detekcja cieni zgodna z prostą fizyką (poprzednia wersja):

    1) BGR + GaussianBlur(5x5)
    2) HSV split
    3) kandydaci: V < v_thresh
    4) odrzucenie ciemnych obiektów: spójność H względem lokalnej średniej (circular)
    5) morfologia: close -> open
    6) filtrowanie komponentów: area + elongation
    7) walidacja kierunkowa: Canny+Hough w ROI komponentu (min_line_conf)
    8) maska binarna 0/255

    Zwraca maskę; gdy debug=True zwraca (mask, dbg_dict).
    """
    if image_bgr is None:
        raise ValueError('Pusty obraz')

    img = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # CLAHE na luminancji (V) — podbija lokalny kontrast bez zmiany H/S.
    if use_clahe:
        try:
            tg = (int(clahe_tile_grid_size[0]), int(clahe_tile_grid_size[1]))
            tg = (max(2, tg[0]), max(2, tg[1]))
        except Exception:
            tg = (8, 8)
        clip = float(clahe_clip_limit) if clahe_clip_limit is not None else 2.0
        clip = max(0.5, min(10.0, clip))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tg)
        v = clahe.apply(v)

    # 3) kandydaci na cień (jasność)
    v_mask = (v < int(v_thresh)).astype(np.uint8) * 255

    # 4) spójność koloru: lokalna średnia H (circular mean sin/cos)
    k = int(max(3, hue_window))
    if k % 2 == 0:
        k += 1
    k = min(k, 101)

    ang = (h.astype(np.float32) / 180.0) * (2.0 * np.pi)
    sin_h = np.sin(ang)
    cos_h = np.cos(ang)
    sin_m = cv2.GaussianBlur(sin_h, (k, k), 0)
    cos_m = cv2.GaussianBlur(cos_h, (k, k), 0)
    mean_ang = np.arctan2(sin_m, cos_m)
    mean_h = ((mean_ang % (2.0 * np.pi)) * (180.0 / (2.0 * np.pi))).astype(np.uint8)

    hue_diff = _circular_hue_diff(h, mean_h)
    hue_ok = (hue_diff <= int(hue_diff_max)).astype(np.uint8) * 255

    cand = cv2.bitwise_and(v_mask, hue_ok)

    # 5) morfologia
    mk = int(max(3, morph_kernel))
    if mk % 2 == 0:
        mk += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kernel)

    # 6) komponenty + elongation
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out = np.zeros_like(cand)
    dbg_regions = []

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h0 = int(stats[i, cv2.CC_STAT_HEIGHT])

        elong = max(w, h0) / (min(w, h0) + 1e-6)
        if elong < float(min_elongation):
            continue

        region_mask = (labels[y:y + h0, x:x + w] == i).astype(np.uint8) * 255
        line_angle = None
        line_conf = 0.0

        # 7) geometria jako filtr binarny
        if use_geometry_validation:
            roi_v = v[y:y + h0, x:x + w]
            edges = cv2.Canny(roi_v, 50, 150)
            edges = cv2.bitwise_and(edges, region_mask)

            line_angle, line_conf = _dominant_line_angle_deg(
                edges,
                min_line_length=max(20, min(w, h0) // 3),
                max_line_gap=10,
                hough_thresh=40,
            )
            if line_conf < float(min_line_conf):
                continue

        out[y:y + h0, x:x + w] = cv2.bitwise_or(out[y:y + h0, x:x + w], region_mask)

        if debug:
            dbg_regions.append({
                'area': area,
                'bbox': (x, y, w, h0),
                'elongation': float(elong),
                'line_angle_deg': None if line_angle is None else float(line_angle),
                'line_conf': float(line_conf),
            })

    if not debug:
        return out

    dbg = {
        'v_mask': v_mask,
        'hue_ok': hue_ok,
        'cand_after_morph': cand,
        'regions': dbg_regions,
    }
    return out, dbg


# Backward-compat alias (tymczasowo), żeby reszta repo się nie wywaliła.
# Nowy kod powinien używać detect_shadow_physics.

def detect_shadow(image_bgr, **kwargs):
    return detect_shadow_physics(image_bgr, **kwargs)
