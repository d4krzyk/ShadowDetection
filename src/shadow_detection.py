import cv2
import numpy as np


def _circular_hue_diff(h1, h2):
    """Różnica H w HSV (OpenCV: H 0..179) z uwzględnieniem cykliczności."""
    d = np.abs(h1.astype(np.int16) - h2.astype(np.int16))
    d = np.minimum(d, 180 - d)
    return d.astype(np.uint8)


def _dominant_line_angle_deg(edge_img, min_line_length=40, max_line_gap=10, hough_thresh=60, return_lines=False):
    """Zwraca (angle_deg, confidence) z HoughLinesP.

    angle_deg w zakresie [0,180) opisuje orientację linii (nie wektora).
    confidence 0..1 mówi jak spójne są kąty (ważone długością).
    """
    lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180.0, threshold=int(hough_thresh),
                            minLineLength=int(min_line_length), maxLineGap=int(max_line_gap))
    if lines is None or len(lines) == 0:
        return (None, 0.0, None) if return_lines else (None, 0.0)

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
        return (None, 0.0, lines) if return_lines else (None, 0.0)

    angs = np.array(angs, dtype=np.float32)
    wts = np.array(wts, dtype=np.float32)

    c = float(np.sum(np.cos(2 * angs) * wts))
    s = float(np.sum(np.sin(2 * angs) * wts))
    mean = 0.5 * np.arctan2(s, c)
    mean = float(mean % np.pi)

    R = ((c * c + s * s) ** 0.5) / (float(np.sum(wts)) + 1e-6)
    conf = float(np.clip(R, 0.0, 1.0))

    if return_lines:
        return float(np.degrees(mean)), conf, lines
    return float(np.degrees(mean)), conf


def _compute_ratio_mask(v8, blur_sizes=(15, 31, 61), ratio_thresholds=(0.75, 0.70, 0.65)):
    """Multi-scale V / blur(V) ratio mask. Returns (mask, ratio_maps, ratio_min)."""
    ratio_maps = []
    masks = []
    v = v8.astype(np.float32) + 1e-6
    for k, thr in zip(blur_sizes, ratio_thresholds):
        kk = int(max(3, k))
        if kk % 2 == 0:
            kk += 1
        bg = cv2.medianBlur(v8, kk).astype(np.float32) + 1e-6
        ratio = v / bg
        ratio_maps.append(ratio)
        masks.append((ratio < float(thr)).astype(np.uint8) * 255)

    if not masks:
        return np.zeros_like(v8), [], None

    ratio_mask = masks[0]
    for m in masks[1:]:
        ratio_mask = cv2.bitwise_or(ratio_mask, m)

    ratio_min = ratio_maps[0]
    for r in ratio_maps[1:]:
        ratio_min = np.minimum(ratio_min, r)

    return ratio_mask, ratio_maps, ratio_min


def detect_shadow_physics(
        image_bgr,
        v_thresh=75,
        hue_window=31,
        hue_diff_max=10,
        min_area=200,
        min_elongation=1.3,
        morph_kernel=5,
        # local contrast (luminance) enhancement
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        # ratio thresholds (multi-scale V/blur(V))
        ratio_blur_sizes=(15, 31, 61),
        ratio_thresholds=(0.75, 0.70, 0.65),
        # soft grayscale shadow score
        use_soft_shadow=True,
        soft_shadow_thresh=0.20,
        soft_shadow_scale=0.40,
        # geometry validation (binary filter)
        use_geometry_validation=True,
        min_line_conf=0.35,
        debug=False,
):
    """Detekcja cieni na podstawie luminancji i spojnosc koloru.

    Zwraca maskę 0/255; gdy debug=True zwraca (mask, dbg_dict).
    """
    if image_bgr is None:
        raise ValueError('Pusty obraz')

    img = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # CLAHE na luminancji (V) — podbija lokalny kontrast bez zmiany H/S.
    if use_clahe:
        # lokalny kontrast tylko na V
        try:
            tg = (int(clahe_tile_grid_size[0]), int(clahe_tile_grid_size[1]))
            tg = (max(2, tg[0]), max(2, tg[1]))
        except Exception:
            tg = (8, 8)
        clip = float(clahe_clip_limit) if clahe_clip_limit is not None else 2.0
        clip = max(0.5, min(10.0, clip))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tg)
        v = clahe.apply(v)

    v8 = v

    # kandydaci cienia: V i V/blur(V)
    v_mask = (v8 < int(v_thresh)).astype(np.uint8) * 255
    ratio_mask, ratio_maps, ratio_min = _compute_ratio_mask(v8, ratio_blur_sizes, ratio_thresholds)

    soft_mask = None
    shadow_score = None
    if use_soft_shadow and ratio_min is not None:
        # shadow_score 0..1, higher = darker relative to local background
        shadow_score = np.clip((1.0 - ratio_min) / float(max(1e-3, soft_shadow_scale)), 0.0, 1.0)
        soft_mask = (shadow_score >= float(soft_shadow_thresh)).astype(np.uint8) * 255
        base_mask = cv2.bitwise_or(v_mask, ratio_mask)
        base_mask = cv2.bitwise_or(base_mask, soft_mask)
    else:
        base_mask = cv2.bitwise_or(v_mask, ratio_mask)

    # spojnosc barwy (H) wzgledem lokalnej sredniej
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

    cand = cv2.bitwise_and(base_mask, hue_ok)

    # morfologia
    mk = int(max(3, morph_kernel))
    if mk % 2 == 0:
        mk += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kernel)

    # komponenty + elongation
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out = np.zeros_like(cand)
    dbg_regions = []
    dbg_lines = []
    weight_map = np.zeros_like(cand, dtype=np.float32)

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
        line_weight = 0.0

        # 7) geometria jako waga, nie filtr
        if use_geometry_validation:
            # Hough jako waga pewnosci kierunku w regionie
            roi_v = v8[y:y + h0, x:x + w]
            edges = cv2.Canny(roi_v, 50, 150)
            edges = cv2.bitwise_and(edges, region_mask)

            if debug:
                line_angle, line_conf, lines = _dominant_line_angle_deg(
                    edges,
                    min_line_length=max(20, min(w, h0) // 3),
                    max_line_gap=10,
                    hough_thresh=40,
                    return_lines=True,
                )
                if lines is not None:
                    for l in lines[:, 0, :]:
                        x1, y1, x2, y2 = [int(vv) for vv in l]
                        dbg_lines.append((x1 + x, y1 + y, x2 + x, y2 + y))
            else:
                line_angle, line_conf = _dominant_line_angle_deg(
                    edges,
                    min_line_length=max(20, min(w, h0) // 3),
                    max_line_gap=10,
                    hough_thresh=40,
                )

        if float(min_line_conf) > 1e-6:
            line_weight = float(np.clip(float(line_conf) / float(min_line_conf), 0.0, 1.0))
        else:
            line_weight = float(np.clip(float(line_conf), 0.0, 1.0))

        out[y:y + h0, x:x + w] = cv2.bitwise_or(out[y:y + h0, x:x + w], region_mask)
        weight_map[y:y + h0, x:x + w] = np.maximum(weight_map[y:y + h0, x:x + w], float(line_weight))

        if debug:
            dbg_regions.append({
                'area': area,
                'bbox': (x, y, w, h0),
                'elongation': float(elong),
                'line_angle_deg': None if line_angle is None else float(line_angle),
                'line_conf': float(line_conf),
                'line_weight': float(line_weight),
            })

    if not debug:
        return out

    # debug maps
    v_f = v8.astype(np.float32)
    gx = cv2.Sobel(v_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(v_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    chroma_diff = None
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_blur = cv2.GaussianBlur(lab, (31, 31), 0)
        ab = lab[:, :, 1:3] / 255.0
        ab_blur = lab_blur[:, :, 1:3] / 255.0
        chroma_diff = np.linalg.norm(ab - ab_blur, axis=2)
    except Exception:
        chroma_diff = None

    dbg = {
        'v_mask': v_mask,
        'ratio_mask': ratio_mask,
        'ratio_maps': ratio_maps,
        'ratio_min': ratio_min,
        'soft_mask': soft_mask,
        'shadow_score': shadow_score,
        'hue_ok': hue_ok,
        'cand_after_morph': cand,
        'grad_mag': grad_mag,
        'chroma_diff': chroma_diff,
        'regions': dbg_regions,
        'lines': dbg_lines,
        'weight_map': weight_map,
    }
    return out, dbg

