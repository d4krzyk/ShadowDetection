import cv2
import numpy as np

from src.shadow_direction import estimate_light_direction_shadow_edges, estimate_light_direction_bgr


def _normalize(v):
    v = np.array(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return v
    return v / n


def estimate_camera_from_vanishing_points(image_bgr, debug=False):
    """Bardzo uproszczona estymacja kamery (K,R) z linii sceny.

    Uwaga: To jest heurystyka do pseudo-3D.

    Założenia:
    - principal point ~ środek obrazu
    - ogniskowa f ~ 1.2 * max(w,h)
    - wykrywamy dominantę linii pionowych i horyzontalnych; próbujemy znaleźć vanishing point horyzontu

    Zwraca: (K, R, t, confidence, dbg)
    - R: przybliżona orientacja (świat: Z-up)
    - t: ustawiamy kamerę na wysokości 1.6 jednostki (umowne)

    Jeśli estymacja się nie uda -> zwraca prostą kamerę patrzącą w dół osi -Z z niską pewnością.
    """
    h, w = image_bgr.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    f = 1.2 * max(w, h)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80, minLineLength=max(40, w // 20), maxLineGap=15)

    if lines is None or len(lines) < 10:
        # fallback
        R = np.eye(3, dtype=np.float32)
        t = np.array([0, 0, -1.6], dtype=np.float32)
        return K, R, t, 0.0, {'reason': 'no_lines'} if debug else (K, R, t, 0.0, None)

    # Collect line orientations
    angs = []
    segs = []
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in l]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        theta = np.degrees(np.arctan2(dy, dx))
        angs.append(theta)
        segs.append((x1, y1, x2, y2))

    angs = np.array(angs, dtype=np.float32)

    # classify near-vertical and near-horizontal
    vert_mask = np.abs(((angs + 90) % 180) - 90) < 15
    hor_mask = np.abs((angs % 180)) < 15

    # if no verticals, weak
    conf = float(np.clip((vert_mask.sum() + hor_mask.sum()) / max(1, len(angs)), 0.0, 1.0))

    # We will assume world Z projects roughly to image vertical direction.
    # That implies camera roll ~ 0. Estimate roll from vertical lines.
    if vert_mask.sum() >= 3:
        # mean angle of vertical lines (should be ~90 or -90)
        vangs = angs[vert_mask]
        # map to -90..90
        vangs_n = ((vangs + 90) % 180) - 90
        roll = float(np.median(vangs_n))
    else:
        roll = 0.0
        conf *= 0.6

    # Build R as rotation around Z (roll correction) only; pitch/yaw unknown.
    r = np.deg2rad(-roll)
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]], dtype=np.float32)

    R = Rz
    t = np.array([0, 0, -1.6], dtype=np.float32)

    dbg = {'roll_deg': roll, 'num_lines': int(len(angs)), 'vert': int(vert_mask.sum()), 'hor': int(hor_mask.sum())}
    return K, R, t, conf, (dbg if debug else None)


def estimate_light_vector_3d(image_bgr, debug=False):
    """Estymuje pseudo-3D wektor światła w układzie świata (Z-up).

    Kroki:
    - camera: K,R,t (heurystycznie)
    - 2D direction: z edge+Hough (shadow edges) z confidence
    - Elevation angle: heurystyka z kontrastu cienia + 'pewności' (im mocniejsze cienie, tym niższe słońce)

    Zwraca:
    - light_dir_world (3,)
    - confidence_total (0..1)
    - dbg
    """
    K, R, t, cam_conf, cam_dbg = estimate_camera_from_vanishing_points(image_bgr, debug=True)

    a2, vec2, c2 = estimate_light_direction_shadow_edges(image_bgr)
    if c2 <= 0.05:
        a2, vec2 = estimate_light_direction_bgr(image_bgr)
        c2 = 0.05

    vx, vy = vec2
    v2 = _normalize([vx, vy])

    # Shadow direction on ground is opposite to light projection; depends on convention.
    # We'll treat (vx,vy) as light direction in image plane; map to world ground plane X,Y.
    # Without full calibration, we align world X,Y with image x,y after roll compensation.
    v_xy_world = _normalize((R.T @ np.array([v2[0], v2[1], 0.0], dtype=np.float32))[:2])

    # Elevation heuristic: estimate "shadow strength" via ratio map dispersion
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    bg = cv2.medianBlur(hsv[:, :, 2], 31).astype(np.float32) + 1e-6
    ratio = (v + 1e-6) / bg
    # stronger shadows -> lower ratio in shadow: use lower percentile
    p10 = float(np.percentile(ratio, 10))
    strength = float(np.clip((1.0 - p10) / 0.35, 0.0, 1.0))  # 0..1

    # Map strength to elevation (in radians). Stronger shadows -> lower sun -> smaller elevation.
    # clamp between 15..70 degrees
    elev_deg = 70.0 - 55.0 * strength
    elev_deg = float(np.clip(elev_deg, 15.0, 70.0))
    elev = np.deg2rad(elev_deg)

    # Build 3D light dir: horizontal magnitude cos(elev), vertical component sin(elev)
    lx = float(v_xy_world[0]) * np.cos(elev)
    ly = float(v_xy_world[1]) * np.cos(elev)
    lz = float(np.sin(elev))

    light_dir_world = _normalize([lx, ly, lz])

    conf = float(np.clip(0.5 * cam_conf + 0.5 * c2, 0.0, 1.0))
    dbg = {
        'cam_conf': cam_conf,
        'edge_conf': c2,
        'elev_deg': elev_deg,
        'shadow_strength': strength,
        'cam_dbg': cam_dbg,
        'angle2d': a2,
    }

    return light_dir_world, conf, (dbg if debug else None), (K, R, t)

