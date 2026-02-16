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
    thr = np.percentile(mag, 90)
    mask = mag >= max(thr, 5.0)
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
    R = (c * c + s * s) ** 0.5 / (np.sum(wts) + 1e-6)
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
