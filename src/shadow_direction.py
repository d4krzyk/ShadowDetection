import cv2
import numpy as np


def _largest_connected_component(mask_u8):
    """Zwraca maskę największej składowej (0/255)."""
    if mask_u8 is None:
        return None

    m = mask_u8
    if len(m.shape) == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    if m.dtype != np.uint8:
        mm = m.astype(np.float32)
        if mm.max() <= 1.0:
            mm *= 255.0
        m = np.clip(mm, 0, 255).astype(np.uint8)

    if int(np.count_nonzero(m)) == 0:
        return np.zeros_like(m)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return (m > 0).astype(np.uint8) * 255

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    return (labels == idx).astype(np.uint8) * 255


def _normalize2(v):
    v = np.array(v, dtype=np.float32).reshape(2)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return v / n


def merge_vectors(components):
    """Laczy wektory kierunku z wagami confidence.

    components: lista dictow z kluczami 'vec' i 'conf'.
    Zwraca: (vec_xy, confidence)
    """
    if not components:
        return (1.0, 0.0), 0.0

    vecs = []
    wts = []
    for c in components:
        if not c:
            continue
        v = c.get('vec')
        w = float(c.get('conf', 0.0))
        if v is None or w <= 1e-6:
            continue
        vecs.append(_normalize2(v))
        wts.append(w)

    if not vecs:
        return (1.0, 0.0), 0.0

    ref = vecs[0]
    aligned = [(-v if float(np.dot(v, ref)) < 0 else v) for v in vecs]

    V = np.vstack(aligned)
    W = np.array(wts, dtype=np.float32).reshape(-1, 1)
    v_final = _normalize2(np.sum(V * W, axis=0))

    dots = [float(np.dot(v_final, v)) for v in aligned]
    agree = float(np.clip((np.mean(dots) + 1.0) / 2.0, 0.0, 1.0))
    avg_conf = float(np.clip(float(np.mean(wts)), 0.0, 1.0))
    conf = float(np.clip(agree * avg_conf, 0.0, 1.0))

    return (float(v_final[0]), float(v_final[1])), float(conf)




def estimate_light_direction_bgr(image_bgr):
    """Szacuje kierunek światła na podstawie dominujacego gradientu jasnosci.

    Zwraca:
    - angle_deg: kąt 0..360 (0=→, 90=↓)
    - vec: (vx, vy) jednostkowy wektor
    """
    if image_bgr is None:
        raise ValueError("Pusty obraz")

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)

    gx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)

    mag = cv2.magnitude(gx, gy)
    ang = cv2.phase(gx, gy, angleInDegrees=True)

    thr = float(np.percentile(mag, 90))
    thr2 = thr if thr > 5.0 else 5.0
    mask = mag >= thr2
    if mask.sum() < 50:
        return 0.0, (1.0, 0.0)

    ang_sel = ang[mask]
    mag_sel = mag[mask]

    bins = 36
    hist, edges = np.histogram(ang_sel, bins=bins, range=(0.0, 360.0), weights=mag_sel)
    best_bin = int(hist.argmax())
    best_angle = (edges[best_bin] + edges[best_bin + 1]) / 2.0

    angle_rad = np.deg2rad(best_angle)
    vx = float(np.cos(angle_rad))
    vy = float(np.sin(angle_rad))

    return float(best_angle % 360.0), (vx, vy)


def estimate_light_direction_shadow_edges(image_bgr,
                                         canny1=60,
                                         canny2=150,
                                         chroma_diff_max=0.10,
                                         v_drop_min=0.10,
                                         hough_thresh=60,
                                         min_line_len=40,
                                         max_line_gap=10):
    """Szacunek kierunku światła na podstawie krawędzi cieni.

    Zwraca:
    - angle_deg (0..360)
    - vec_xy (vx, vy)
    - confidence (0..1)
    """
    if image_bgr is None:
        raise ValueError('Pusty obraz')

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    v8 = hsv[:, :, 2]

    bg = cv2.medianBlur(v8, 31).astype(np.float32) + 1e-6
    ratio = (v + 1e-6) / bg
    shadowish = ratio < (1.0 - float(v_drop_min))

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_blur = cv2.GaussianBlur(lab, (31, 31), 0)
    ab = lab[:, :, 1:3] / 255.0
    ab_blur = lab_blur[:, :, 1:3] / 255.0
    chroma_diff = np.linalg.norm(ab - ab_blur, axis=2)
    chroma_ok = chroma_diff < float(chroma_diff_max)

    gx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)
    g_thr = np.percentile(gmag, 90)
    grad_ok = gmag >= max(5.0, g_thr)

    boundary_like = (shadowish & chroma_ok & grad_ok)

    edges = cv2.Canny(v8, int(canny1), int(canny2))
    edges = cv2.bitwise_and(edges, (boundary_like.astype(np.uint8) * 255))

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=int(hough_thresh),
                            minLineLength=int(min_line_len), maxLineGap=int(max_line_gap))

    if lines is None or len(lines) == 0:
        ang, vec = estimate_light_direction_bgr(image_bgr)
        return ang, vec, 0.0

    angs = []
    wts = []
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(t) for t in l]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1.0:
            continue
        theta = np.arctan2(dy, dx)
        angs.append(theta)
        wts.append(length)

    if not angs:
        ang, vec = estimate_light_direction_bgr(image_bgr)
        return ang, vec, 0.0

    angs = np.array(angs, dtype=np.float32)
    wts = np.array(wts, dtype=np.float32)

    c = np.sum(np.cos(2 * angs) * wts)
    s = np.sum(np.sin(2 * angs) * wts)
    mean_line = 0.5 * np.arctan2(s, c)

    R = float(((c * c + s * s) ** 0.5) / (float(np.sum(wts)) + 1e-6))
    confidence = float(np.clip(R, 0.0, 1.0))

    cand1 = mean_line + np.pi / 2.0
    cand2 = mean_line - np.pi / 2.0

    ang_g, vec_g = estimate_light_direction_bgr(image_bgr)
    vg = np.array(vec_g, dtype=np.float32)

    v1 = np.array([np.cos(cand1), np.sin(cand1)], dtype=np.float32)
    v2 = np.array([np.cos(cand2), np.sin(cand2)], dtype=np.float32)

    if float(np.dot(v1, vg)) >= float(np.dot(v2, vg)):
        v_best = v1
    else:
        v_best = v2

    angle_deg = (np.degrees(np.arctan2(float(v_best[1]), float(v_best[0]))) + 360.0) % 360.0
    vec_xy = (float(v_best[0]), float(v_best[1]))

    return float(angle_deg), vec_xy, confidence


def estimate_light_direction_shadow_edge_gradient(
        image_bgr,
        shadow_mask,
        weight_map=None,
        edge_ring=2,
        mag_percentile=75,
        hp_blur_ksize=51,
        debug=False,
):
    """Kierunek z gradientu luminancji liczony tylko na krawędzi cienia.

    To zwykle stabilniejsze niż globalny gradient, bo ograniczamy się do miejsca,
    gdzie cień faktycznie zmienia jasność.

    Zwraca: (angle_deg, vec_xy, confidence) lub +dbg gdy debug=True.
    """
    if image_bgr is None or shadow_mask is None:
        out = (0.0, (1.0, 0.0), 0.0)
        return (*out, {'reason': 'missing'}) if debug else out

    m = shadow_mask
    if len(m.shape) == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.dtype != np.uint8:
        mm = m.astype(np.float32)
        if mm.max() <= 1.0:
            mm *= 255.0
        m = np.clip(mm, 0, 255).astype(np.uint8)

    if int(np.count_nonzero(m)) < 50:
        out = (0.0, (1.0, 0.0), 0.0)
        return (*out, {'reason': 'too_few_mask_pixels'}) if debug else out

    # Inner-ring: wąski pas *wewnątrz* cienia (zmniejsza wpływ tekstury jasnej strony).
    k = int(max(1, edge_ring))
    k = min(k, 9)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    m_bin = (m > 0).astype(np.uint8) * 255
    inner = cv2.erode(m_bin, ker, iterations=1)
    ring = cv2.subtract(m_bin, inner)
    edge = (ring > 0)

    if int(np.count_nonzero(edge)) < 30:
        out = (0.0, (1.0, 0.0), 0.0)
        return (*out, {'reason': 'too_few_edge_pixels'}) if debug else out

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)

    # Tłumienie tekstury: high-pass na V (V - blur(V)).
    kk = int(max(3, hp_blur_ksize))
    if kk % 2 == 0:
        kk += 1
    kk = min(kk, 151)
    v_blur = cv2.GaussianBlur(v, (kk, kk), 0)
    v_hp = v - v_blur

    gx = cv2.Sobel(v_hp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(v_hp, cv2.CV_32F, 0, 1, ksize=3)

    ys, xs = np.where(edge)
    gxs = gx[ys, xs]
    gys = gy[ys, xs]
    mag = np.sqrt(gxs * gxs + gys * gys) + 1e-6

    # Odrzucenie slabego gradientu (najczesciej faktura / szum).
    mp = float(np.clip(float(mag_percentile), 0.0, 99.0))
    thr = float(np.percentile(mag, mp))
    keep = mag >= max(1e-6, thr)
    if int(np.count_nonzero(keep)) < 20:
        keep = mag >= max(1e-6, float(np.percentile(mag, 60.0)))

    # wagi: siła gradientu * opcjonalna waga z mapy (np. shadow_score 0..1)
    w = mag.copy()
    if weight_map is not None:
        wm = weight_map
        if len(wm.shape) == 3:
            wm = cv2.cvtColor(wm, cv2.COLOR_BGR2GRAY)
        if wm.shape[:2] != v.shape[:2]:
            wm = cv2.resize(wm, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_LINEAR)
        wm = wm.astype(np.float32)
        if float(wm.max()) > 1.5:
            wm = wm / 255.0
        w = w * np.clip(wm[ys, xs], 0.0, 1.0)

    # zastosuj keep
    w = w[keep]
    vx = (gxs / mag)[keep]
    vy = (gys / mag)[keep]

    sx = float(np.sum(vx * w))
    sy = float(np.sum(vy * w))
    norm = float((sx * sx + sy * sy) ** 0.5)

    if norm < 1e-6:
        out = (0.0, (1.0, 0.0), 0.0)
        return (*out, {'reason': 'zero_resultant'}) if debug else out

    v_out = (sx / norm, sy / norm)

    ang = (np.degrees(np.arctan2(v_out[1], v_out[0])) + 360.0) % 360.0

    # confidence = spójność kierunków na krawędzi
    conf = float(np.clip(norm / (float(np.sum(w)) + 1e-6), 0.0, 1.0))
    # penalizuj, gdy zostalo malo punktow (mniej wiarygodne)
    size_boost = float(np.clip((float(vx.size) - 50.0) / 500.0, 0.0, 1.0))
    conf = float(np.clip(0.15 + 0.85 * conf, 0.0, 1.0))
    conf = float(np.clip(conf * (0.6 + 0.4 * size_boost), 0.0, 1.0))

    if not debug:
        return float(ang), (float(v_out[0]), float(v_out[1])), float(conf)

    dbg = {
        'ring_pixels': int(np.count_nonzero(edge)),
        'kept_pixels': int(vx.size),
        'mag_percentile': float(mp),
        'mag_thr': float(thr),
        'hp_blur_ksize': int(kk),
        'sum_w': float(np.sum(w)),
        'norm': float(norm),
    }
    return float(ang), (float(v_out[0]), float(v_out[1])), float(conf), dbg


def estimate_from_mask_pca_tip(shadow_mask, weight_map=None, min_pixels=300, tip_quantile=0.92, debug=False):
    """PCA na masce cienia (największa składowa) z opcjonalnymi wagami szarosci.

    Zwraca wektor od tip -> centroid oraz confidence 0..1.
    """
    if shadow_mask is None:
        raise ValueError('Pusta maska')

    if len(shadow_mask.shape) == 3:
        m = cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY)
    else:
        m = shadow_mask

    if m.dtype != np.uint8:
        mm = m.astype(np.float32)
        if mm.max() <= 1.0:
            mm = mm * 255.0
        m = np.clip(mm, 0, 255).astype(np.uint8)

    m = _largest_connected_component(m)

    pts = np.column_stack(np.where(m > 0))
    if pts.shape[0] < int(min_pixels):
        v = np.array([1.0, 0.0], dtype=np.float32)
        return (v, 0.0, {'reason': 'too_few_pixels', 'n': int(pts.shape[0])}) if debug else (v, 0.0)

    xy = pts[:, ::-1].astype(np.float32)  # (x,y)

    weights = None
    if weight_map is not None:
        if len(weight_map.shape) == 3:
            wmap = cv2.cvtColor(weight_map, cv2.COLOR_BGR2GRAY)
        else:
            wmap = weight_map
        w = wmap[pts[:, 0], pts[:, 1]].astype(np.float32)
        if w.max() > 1.0:
            w = w / 255.0
        w = np.clip(w, 0.0, 1.0)
        if float(np.sum(w)) > 1e-6:
            weights = w

    if weights is None:
        mean = xy.mean(axis=0)
        X = xy - mean
        cov = (X.T @ X) / float(max(1, X.shape[0] - 1))
    else:
        wsum = float(np.sum(weights))
        mean = (xy * weights[:, None]).sum(axis=0) / max(1e-6, wsum)
        X = xy - mean
        cov = (X.T * weights) @ X / max(1e-6, wsum)

    evals, evecs = np.linalg.eigh(cov)  # evals asc
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    main_axis = evecs[:, 0].astype(np.float32)
    main_axis = main_axis / (np.linalg.norm(main_axis) + 1e-6)

    proj = (X @ main_axis).astype(np.float32)

    gray_mu = 0.0
    gray_conf = 0.0
    if weights is not None:
        wsum = float(np.sum(weights))
        gray_mu = float(np.sum(weights * proj) / max(1e-6, wsum))
        p_std = float(np.std(proj) + 1e-6)
        gray_conf = float(np.clip(abs(gray_mu) / (2.0 * p_std), 0.0, 1.0))
        if gray_mu < 0.0:
            main_axis = -main_axis
            proj = -proj

    # Tip selection: far end opposite to the weighted (darker) side
    q_far = 1.0 - float(tip_quantile)
    if weights is not None:
        thr = float(np.quantile(proj, q_far))
        tip_pts = xy[proj <= thr]
    else:
        thr = float(np.quantile(proj, float(tip_quantile)))
        tip_pts = xy[proj >= thr]

    if tip_pts.shape[0] < 10:
        tip = xy[int(np.argmin(proj))] if weights is not None else xy[int(np.argmax(proj))]
    else:
        tip = tip_pts.mean(axis=0)

    dir_vec = (mean - tip).astype(np.float32)
    if float(np.linalg.norm(dir_vec)) < 1e-6:
        dir_vec = -main_axis
    dir_vec = _normalize2(dir_vec)

    lam1 = float(max(evals[0], 1e-6))
    lam2 = float(max(evals[1], 1e-6))
    ratio = lam1 / lam2
    elong_conf = float(np.clip((ratio - 1.2) / 4.0, 0.0, 1.0))
    size_conf = float(np.clip((pts.shape[0] - min_pixels) / 5000.0, 0.0, 1.0))
    conf = float(np.clip(0.15 + 0.55 * elong_conf + 0.2 * size_conf + 0.1 * gray_conf, 0.0, 1.0))

    if not debug:
        return dir_vec, conf

    dbg = {
        'centroid_xy': (float(mean[0]), float(mean[1])),
        'tip_xy': (float(tip[0]), float(tip[1])),
        'axis_vec': (float(main_axis[0]), float(main_axis[1])),
        'dir_vec': (float(dir_vec[0]), float(dir_vec[1])),
        'eigs': (float(lam1), float(lam2)),
        'eig_ratio': float(ratio),
        'n_pixels': int(pts.shape[0]),
        'elong_conf': float(elong_conf),
        'size_conf': float(size_conf),
        'gray_mu': float(gray_mu),
        'gray_conf': float(gray_conf),
    }
    return dir_vec, conf, dbg


def is_object_visible_opposite_shadow(image_bgr, shadow_mask, shadow_dir_xy, sample_len=45, band_half=8, dark_drop=18, debug=False):
    """Heurystyka: czy przy nasadzie cienia widac obiekt rzucajacy cien."""
    if image_bgr is None or shadow_mask is None:
        return (False, 0.0, {'reason': 'missing'}) if debug else (False, 0.0)

    if len(shadow_mask.shape) == 3:
        m = cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY)
    else:
        m = shadow_mask

    ys, xs = np.where(m > 0)
    if ys.size < 200:
        return (False, 0.0, {'reason': 'too_few_mask'}) if debug else (False, 0.0)

    cx = float(xs.mean())
    cy = float(ys.mean())

    v = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    h, w = v.shape

    d = _normalize2(shadow_dir_xy)
    # idziemy w stronę obiektu (przeciwnie do cienia)
    ox = cx - float(d[0]) * float(sample_len)
    oy = cy - float(d[1]) * float(sample_len)

    def clip_int(a, lo, hi):
        return int(max(lo, min(hi, round(a))))

    x0 = clip_int(ox, 0, w - 1)
    y0 = clip_int(oy, 0, h - 1)

    # próbkujemy pasek prostopadły do kierunku, żeby złapać obiekt
    perp = np.array([-d[1], d[0]], dtype=np.float32)
    vals = []
    for t in range(-band_half, band_half + 1):
        xi = clip_int(x0 + perp[0] * t, 0, w - 1)
        yi = clip_int(y0 + perp[1] * t, 0, h - 1)
        vals.append(float(v[yi, xi]))
    obj_v = float(np.median(vals))

    # tło: punkt przesunięty jeszcze dalej "za" obiekt (jeszcze bardziej przeciwnie) – powinien być jaśniejszy lub podobny
    bx = cx - float(d[0]) * float(sample_len + 30)
    by = cy - float(d[1]) * float(sample_len + 30)
    xb = clip_int(bx, 0, w - 1)
    yb = clip_int(by, 0, h - 1)
    bg_v = float(v[yb, xb])

    drop = bg_v - obj_v

    # dodatkowo: krawędź w okolicy (gradient)
    gx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    gmag = float(cv2.magnitude(gx, gy)[y0, x0])

    visible = (drop >= float(dark_drop)) or (gmag >= 25.0)

    conf = 0.0
    conf = max(conf, float(np.clip(drop / (float(dark_drop) * 2.0), 0.0, 1.0)))
    conf = max(conf, float(np.clip((gmag - 10.0) / 60.0, 0.0, 1.0)))

    if not debug:
        return bool(visible), float(conf)

    dbg = {
        'centroid_xy': (cx, cy),
        'sample_xy': (x0, y0),
        'bg_xy': (xb, yb),
        'obj_v': obj_v,
        'bg_v': bg_v,
        'drop': drop,
        'gmag': gmag,
    }
    return bool(visible), float(conf), dbg


def fuse_light_vectors(image_bgr, shadow_mask, shadow_score=None, prefer_mask=True, debug=False):
    """Łączy kilka estymatorów kierunku światła (2D) i zwraca finalny wektor + confidence.

    Składniki:
    A) z maski: PCA+tip -> kierunek od tip do centroidu (opcjonalnie z shadow_score)
    B) z obrazu: krawędzie cieni + Hough (estimate_light_direction_shadow_edges)
    C) fallback: gradient (estimate_light_direction_bgr)

    Zwraca:
      angle_deg, vec_xy, confidence, dbg
    """
    # B) Hough on shadow-like edges
    ang_h, vec_h, conf_h = estimate_light_direction_shadow_edges(image_bgr)
    v_h = _normalize2(vec_h)

    # C) gradient fallback
    ang_g, vec_g = estimate_light_direction_bgr(image_bgr)
    v_g = _normalize2(vec_g)

    # A) mask PCA (with optional gray weights)
    if shadow_mask is not None:
        pca_out = estimate_from_mask_pca_tip(shadow_mask, weight_map=shadow_score, debug=True)
        v_m, conf_m, dbg_m = pca_out
        visible, conf_obj, dbg_obj = is_object_visible_opposite_shadow(image_bgr, shadow_mask, _normalize2(-np.array(v_m, dtype=np.float32)), debug=True)
        conf_m2 = float(np.clip(conf_m + 0.25 * conf_obj, 0.0, 1.0))
    else:
        v_m = None
        conf_m2 = 0.0
        dbg_m = None
        visible = False
        conf_obj = 0.0
        dbg_obj = None

    # weights
    w_h = float(np.clip(conf_h, 0.0, 1.0))
    w_m = float(np.clip(conf_m2, 0.0, 1.0))
    w_g = 0.15

    if prefer_mask:
        w_m *= 1.15

    if w_h > 0.55 and w_m < 0.25:
        w_h *= 1.15

    components = [
        {'name': 'hough', 'vec': v_h, 'conf': w_h},
        {'name': 'grad', 'vec': v_g, 'conf': w_g},
    ]
    if v_m is not None:
        components.append({'name': 'mask', 'vec': v_m, 'conf': w_m})

    v_final, conf = merge_vectors(components)
    angle_deg = (np.degrees(np.arctan2(float(v_final[1]), float(v_final[0]))) + 360.0) % 360.0

    if not debug:
        return float(angle_deg), (float(v_final[0]), float(v_final[1])), float(conf)

    dbg = {
        'v_final': (float(v_final[0]), float(v_final[1])),
        'conf_final': float(conf),
        'hough': {'angle': float(ang_h), 'vec': (float(v_h[0]), float(v_h[1])), 'conf': float(conf_h)},
        'grad': {'angle': float(ang_g), 'vec': (float(v_g[0]), float(v_g[1]))},
        'mask': None if dbg_m is None else {'conf': float(conf_m2), 'dbg': dbg_m, 'object_visible': bool(visible), 'obj_conf': float(conf_obj), 'obj_dbg': dbg_obj},
        'weights': {'w_h': w_h, 'w_m': w_m, 'w_g': w_g},
    }
    return float(angle_deg), (float(v_final[0]), float(v_final[1])), float(conf), dbg

