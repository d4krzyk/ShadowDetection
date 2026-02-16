import cv2
import numpy as np


def estimate_light_direction_bgr(image_bgr):
    """Szacuje kierunek światła w obrazie.

    Zwraca:
    - angle_deg: kąt w stopniach (0..360), gdzie 0=→ (oś X dodatnia), 90=↓
    - vec: (vx, vy) jednostkowy wektor kierunku

    Heurystyka: liczymy gradient jasności (Sobel) i bierzemy histogram orientacji gradientu.
    Cień zwykle powoduje granice (spadek V), a kierunek światła jest w przybliżeniu prostopadły
    do dominujących krawędzi cienia. To jest tylko wskazówka (nie zawsze poprawna), ale pomaga.
    """
    if image_bgr is None:
        raise ValueError("Pusty obraz")

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)

    gx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)

    mag = cv2.magnitude(gx, gy)
    ang = cv2.phase(gx, gy, angleInDegrees=True)  # 0..360

    # weź tylko mocne krawędzie
    thr = float(np.percentile(mag, 90))
    thr2 = thr if thr > 5.0 else 5.0
    mask = mag >= thr2
    if mask.sum() < 50:
        # fallback
        return 0.0, (1.0, 0.0)

    ang_sel = ang[mask]
    mag_sel = mag[mask]

    # histogram 36 binów
    bins = 36
    hist, edges = np.histogram(ang_sel, bins=bins, range=(0.0, 360.0), weights=mag_sel)
    best_bin = int(hist.argmax())
    best_angle = (edges[best_bin] + edges[best_bin + 1]) / 2.0

    # Kierunek światła ~ przeciwny do gradientu spadku jasności.
    # Gradient wskazuje kierunek wzrostu jasności, więc światło jest podobne do gradientu.
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

    Pipeline:
    1) Liczymy mapę pikseli "shadow-boundary-like":
       - duży gradient jasności V
       - mała zmiana chromy (Lab a/b) w poprzek krawędzi (w przybliżeniu: lab vs blur)
       - preferujemy spadek V (ciemniej) względem tła (V/medianBlur(V))
    2) Canny na V i maskujemy tylko shadow-like
    3) HoughLinesP na maskowanych krawędziach
    4) Dominująca orientacja linii => kierunek światła to wektor prostopadły do linii

    Zwraca:
    - angle_deg (0..360)
    - vec_xy (vx, vy)
    - confidence (0..1)

    Uwaga: to nadal 2D na płaszczyźnie obrazu.
    """
    if image_bgr is None:
        raise ValueError('Pusty obraz')

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    v8 = hsv[:, :, 2]

    # local background and ratio test (shadow should be darker)
    bg = cv2.medianBlur(v8, 31).astype(np.float32) + 1e-6
    ratio = (v + 1e-6) / bg
    shadowish = ratio < (1.0 - float(v_drop_min))

    # chroma difference small
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_blur = cv2.GaussianBlur(lab, (31, 31), 0)
    ab = lab[:, :, 1:3] / 255.0
    ab_blur = lab_blur[:, :, 1:3] / 255.0
    chroma_diff = np.linalg.norm(ab - ab_blur, axis=2)
    chroma_ok = chroma_diff < float(chroma_diff_max)

    # gradient magnitude on V
    gx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)
    g_thr = np.percentile(gmag, 90)
    grad_ok = gmag >= max(5.0, g_thr)

    boundary_like = (shadowish & chroma_ok & grad_ok)

    # edges
    edges = cv2.Canny(v8, int(canny1), int(canny2))
    edges = cv2.bitwise_and(edges, (boundary_like.astype(np.uint8) * 255))

    # Hough lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=int(hough_thresh),
                            minLineLength=int(min_line_len), maxLineGap=int(max_line_gap))

    if lines is None or len(lines) == 0:
        # fallback to gradient-based method
        ang, vec = estimate_light_direction_bgr(image_bgr)
        return ang, vec, 0.0

    # accumulate orientation of lines weighted by length
    # line angle in image coords: atan2(dy, dx)
    angs = []
    wts = []
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(t) for t in l]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1.0:
            continue
        theta = np.arctan2(dy, dx)  # -pi..pi
        angs.append(theta)
        wts.append(length)

    if not angs:
        ang, vec = estimate_light_direction_bgr(image_bgr)
        return ang, vec, 0.0

    angs = np.array(angs, dtype=np.float32)
    wts = np.array(wts, dtype=np.float32)

    # circular mean for line orientation (mod pi, because line direction is ambiguous)
    # Use doubling trick: angle*2 then mean, then /2.
    c = np.sum(np.cos(2 * angs) * wts)
    s = np.sum(np.sin(2 * angs) * wts)
    mean_line = 0.5 * np.arctan2(s, c)  # -pi/2..pi/2

    # confidence from resultant length
    R = float(((c * c + s * s) ** 0.5) / (float(np.sum(wts)) + 1e-6))
    confidence = float(np.clip(R, 0.0, 1.0))

    # Light direction is perpendicular to line orientation.
    # Choose one of the two perpendiculars by preferring direction that points from darker to brighter
    # Approx: use global gradient direction estimate as tie-breaker.
    cand1 = mean_line + np.pi / 2.0
    cand2 = mean_line - np.pi / 2.0

    # get gradient-based vector as hint
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


def _normalize2(v):
    v = np.array(v, dtype=np.float32).reshape(2)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return v / n


def _circular_mean_pi(angs_rad, wts=None):
    """Średnia kołowa dla orientacji linii (mod pi). Zwraca (mean, R)."""
    angs = np.array(angs_rad, dtype=np.float32).reshape(-1)
    if angs.size == 0:
        return 0.0, 0.0
    if wts is None:
        w = np.ones_like(angs, dtype=np.float32)
    else:
        w = np.array(wts, dtype=np.float32).reshape(-1)
        if w.size != angs.size:
            w = np.ones_like(angs, dtype=np.float32)

    c = float(np.sum(np.cos(2 * angs) * w))
    s = float(np.sum(np.sin(2 * angs) * w))
    mean = 0.5 * float(np.arctan2(s, c))
    R = float(((c * c + s * s) ** 0.5) / (float(np.sum(w)) + 1e-6))
    return float(mean), float(np.clip(R, 0.0, 1.0))


def estimate_from_mask_pca_tip(shadow_mask, min_pixels=300, tip_quantile=0.92, debug=False):
    """Estymuje kierunek cienia (a więc przeciwny do światła) z maski binarnej.

    Idea:
    - PCA daje dominującą oś obszaru cienia (orientacja, bez zwrotu).
    - "Tip" (czubek) wybieramy jako fragment maski najdalej wzdłuż osi PCA.
    - Zwrot osi ustawiamy tak, żeby wektor wskazywał od centroidu do tipa.
    - Confidence rośnie gdy obszar jest wydłużony (ratio eigenvalues) i jest wystarczająco duży.

    Zwraca:
      shadow_dir_xy (2,) jednostkowy wektor (kierunek *cienia* w obrazie)
      confidence (0..1)
      dbg (opcjonalnie)
    """
    if shadow_mask is None:
        raise ValueError('Pusta maska')

    if len(shadow_mask.shape) == 3:
        m = cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY)
    else:
        m = shadow_mask

    pts = np.column_stack(np.where(m > 0))  # (y,x)
    if pts.shape[0] < int(min_pixels):
        v = np.array([1.0, 0.0], dtype=np.float32)
        return (v, 0.0, {'reason': 'too_few_pixels', 'n': int(pts.shape[0])}) if debug else (v, 0.0)

    # PCA na (x,y)
    xy = pts[:, ::-1].astype(np.float32)  # (x,y)
    mean = xy.mean(axis=0)
    X = xy - mean

    cov = (X.T @ X) / float(max(1, X.shape[0] - 1))
    evals, evecs = np.linalg.eigh(cov)  # evals rosnąco
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    main_axis = evecs[:, 0].astype(np.float32)
    main_axis = main_axis / (np.linalg.norm(main_axis) + 1e-6)

    # Tip detection: punkty o największej projekcji na oś
    proj = (X @ main_axis).astype(np.float32)
    thr = float(np.quantile(proj, float(tip_quantile)))
    tip_pts = xy[proj >= thr]
    if tip_pts.shape[0] < 10:
        # fallback: max projection point
        tip = xy[int(np.argmax(proj))]
    else:
        tip = tip_pts.mean(axis=0)

    # zwrot: od centroidu do tipa
    dir_vec = (tip - mean).astype(np.float32)
    if float(np.linalg.norm(dir_vec)) < 1e-6:
        dir_vec = main_axis
    shadow_dir = _normalize2(dir_vec)

    # confidence: elongation via eigen ratio + size factor
    lam1 = float(max(evals[0], 1e-6))
    lam2 = float(max(evals[1], 1e-6))
    ratio = lam1 / lam2
    # map ratio ~1..10+ do 0..1
    elong_conf = float(np.clip((ratio - 1.2) / 4.0, 0.0, 1.0))
    size_conf = float(np.clip((pts.shape[0] - min_pixels) / 5000.0, 0.0, 1.0))
    conf = float(np.clip(0.2 + 0.6 * elong_conf + 0.2 * size_conf, 0.0, 1.0))

    if not debug:
        return shadow_dir, conf

    dbg = {
        'centroid_xy': (float(mean[0]), float(mean[1])),
        'tip_xy': (float(tip[0]), float(tip[1])),
        'axis_vec': (float(main_axis[0]), float(main_axis[1])),
        'shadow_dir': (float(shadow_dir[0]), float(shadow_dir[1])),
        'eigs': (float(lam1), float(lam2)),
        'eig_ratio': float(ratio),
        'n_pixels': int(pts.shape[0]),
        'elong_conf': float(elong_conf),
        'size_conf': float(size_conf),
    }
    return shadow_dir, conf, dbg


def is_object_visible_opposite_shadow(image_bgr, shadow_mask, shadow_dir_xy, sample_len=45, band_half=8, dark_drop=18, debug=False):
    """Heurystyka: czy przy "nasadzie" cienia widać obiekt rzucający cień.

    Pomysł:
    - Bierzemy centroid maski jako przybliżenie środka cienia.
    - Idziemy *przeciwnie* do kierunku cienia (czyli w stronę obiektu) o kilkadziesiąt px.
    - Porównujemy luminancję w pobliżu tej pozycji z luminancją w tle wokół – jeśli jest wyraźny "ciemny" obiekt
      lub silna krawędź, to zakładamy że obiekt jest widoczny.

    Zwraca: visible(bool), confidence(0..1), dbg
    """
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


def fuse_light_vectors(image_bgr, shadow_mask, prefer_mask=True, debug=False):
    """Łączy kilka estymatorów kierunku światła (2D) i zwraca finalny wektor + confidence.

    Składniki:
    A) z maski: PCA+tip -> daje kierunek cienia; światło = -shadow_dir
    B) z obrazu: krawędzie cieni + Hough (estimate_light_direction_shadow_edges)
    C) fallback: gradient (estimate_light_direction_bgr)

    Dodatkowo:
    - heurystyka "czy obiekt widoczny" podbija zaufanie do zwrotu z maski

    Zwraca:
      angle_deg, vec_xy, confidence, dbg
    """
    # B) Hough on shadow-like edges
    ang_h, vec_h, conf_h = estimate_light_direction_shadow_edges(image_bgr)
    v_h = _normalize2(vec_h)

    # C) gradient fallback
    ang_g, vec_g = estimate_light_direction_bgr(image_bgr)
    v_g = _normalize2(vec_g)

    # A) mask PCA
    if shadow_mask is not None:
        pca_out = estimate_from_mask_pca_tip(shadow_mask, debug=True)
        shadow_dir, conf_m, dbg_m = pca_out
        v_m = _normalize2(-np.array(shadow_dir, dtype=np.float32))  # światło przeciwnie do cienia
        visible, conf_obj, dbg_obj = is_object_visible_opposite_shadow(image_bgr, shadow_mask, shadow_dir, debug=True)
        # jeśli obiekt widoczny, podbij confidence maski
        conf_m2 = float(np.clip(conf_m + 0.25 * conf_obj, 0.0, 1.0))
    else:
        v_m = None
        conf_m2 = 0.0
        dbg_m = None
        visible = False
        conf_obj = 0.0
        dbg_obj = None

    # wagi
    w_h = float(np.clip(conf_h, 0.0, 1.0))
    w_m = float(np.clip(conf_m2, 0.0, 1.0))
    w_g = 0.15  # zawsze trochę stabilizacji

    if prefer_mask:
        w_m *= 1.15

    # jeśli hough mocny, a maska słaba -> zwiększ hough
    if w_h > 0.55 and w_m < 0.25:
        w_h *= 1.15

    # sumowanie wektorów, ale uwaga na znak: jeśli v_m jest przeciwny do v_h, wybierz zgodny znak
    vecs = []
    wts = []

    def add_vec(v, w):
        if v is None or w <= 1e-6:
            return
        vecs.append(_normalize2(v))
        wts.append(float(w))

    add_vec(v_h, w_h)
    add_vec(v_g, w_g)
    add_vec(v_m, w_m)

    if not vecs:
        v_final = np.array([1.0, 0.0], dtype=np.float32)
        conf = 0.0
    else:
        # align signs to first vector
        ref = vecs[0]
        aligned = []
        for v in vecs:
            if float(np.dot(v, ref)) < 0:
                aligned.append(-v)
            else:
                aligned.append(v)
        V = np.vstack(aligned)
        W = np.array(wts, dtype=np.float32).reshape(-1, 1)
        v_final = _normalize2(np.sum(V * W, axis=0))

        # spójność: średnia zgodność
        dots = [float(np.dot(v_final, v)) for v in aligned]
        agree = float(np.clip((np.mean(dots) + 1.0) / 2.0, 0.0, 1.0))

        # confidence końcowe: waga * spójność
        w_sum = float(np.sum(wts))
        w_norm = float(np.clip(w_sum / 1.6, 0.0, 1.0))
        conf = float(np.clip(agree * (0.25 + 0.75 * w_norm), 0.0, 1.0))

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
