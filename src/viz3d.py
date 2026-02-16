import cv2
import numpy as np


def _project_points(points_xyz, K, R, t):
    """Project Nx3 world points to image using pinhole camera."""
    P = K @ np.hstack([R, t.reshape(3, 1)])  # 3x4
    pts_h = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float32)])
    cam = (P @ pts_h.T).T
    z = cam[:, 2:3] + 1e-6
    uv = cam[:, :2] / z
    return uv


def draw_3d_grid_and_vector(img_bgr, K, R, t, light_dir_world,
                            grid_size=6, grid_step=1.0, color_grid=(80, 80, 80), color_axes=(0, 255, 0), color_vec=(0, 255, 255)):
    """Rysuje prostą siatkę 3D (płaszczyzna gruntu) i wektor światła w układzie świata.

    Układ świata:
    - płaszczyzna gruntu: Z=0
    - oś Z: do góry
    - oś X/Y: na gruncie

    To jest wizualizacja *pseudo-3D* — sensowna jeśli K,R,t są sensownie oszacowane.
    """
    if img_bgr is None:
        return img_bgr

    out = img_bgr

    # grid points/lines on Z=0
    half = int(grid_size)
    lines = []
    for i in range(-half, half + 1):
        # lines parallel to X
        p1 = np.array([[-half * grid_step, i * grid_step, 0.0], [half * grid_step, i * grid_step, 0.0]], dtype=np.float32)
        lines.append((p1, color_grid, 1))
        # lines parallel to Y
        p2 = np.array([[i * grid_step, -half * grid_step, 0.0], [i * grid_step, half * grid_step, 0.0]], dtype=np.float32)
        lines.append((p2, color_grid, 1))

    # axes
    axes = [
        (np.array([[0, 0, 0], [2 * grid_step, 0, 0]], dtype=np.float32), (0, 0, 255), 2),  # X red
        (np.array([[0, 0, 0], [0, 2 * grid_step, 0]], dtype=np.float32), (0, 255, 0), 2),  # Y green
        (np.array([[0, 0, 0], [0, 0, 2 * grid_step]], dtype=np.float32), (255, 0, 0), 2),  # Z blue
    ]

    def draw_segments(segments, col, thick):
        uv = _project_points(segments, K, R, t)
        p1 = tuple(np.round(uv[0]).astype(int))
        p2 = tuple(np.round(uv[1]).astype(int))
        # draw if inside some margin
        cv2.line(out, p1, p2, col, thick, cv2.LINE_AA)

    for seg, col, thick in lines:
        draw_segments(seg, col, thick)
    for seg, col, thick in axes:
        draw_segments(seg, col, thick)

    # light vector from origin in world
    v = np.array(light_dir_world, dtype=np.float32)
    vn = v / (np.linalg.norm(v) + 1e-6)
    start = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    end = start + vn.reshape(1, 3) * (3.0 * grid_step)

    uv = _project_points(np.vstack([start, end]), K, R, t)
    p1 = tuple(np.round(uv[0]).astype(int))
    p2 = tuple(np.round(uv[1]).astype(int))
    cv2.arrowedLine(out, p1, p2, color_vec, 2, tipLength=0.15)

    return out


def render_light_cube_widget(light_dir_world,
                             size=120,
                             yaw_deg=-35.0,
                             pitch_deg=25.0,
                             roll_deg=0.0,
                             bg_color=(20, 20, 20),
                             line_color=(200, 200, 200),
                             axis_colors=((0, 0, 255), (0, 255, 0), (255, 0, 0)),
                             vec_color=(0, 255, 255)):
    """Renderuje małą kostkę 3D z osiami i strzałką kierunku światła.

    Zwraca obraz BGR (size x size).

    Układ widgetu:
    - osie świata: X (czerwony), Y (zielony), Z (niebieski)
    - strzałka: light_dir_world (jednostkowy wektor)

    Render jest niezależny od obrazu sceny (to tylko ikona/kontrolka).
    """
    size = int(max(64, size))
    canvas = np.full((size, size, 3), bg_color, dtype=np.uint8)

    # Simple camera for widget
    f = 2.2 * size
    cx = cy = (size - 1) / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)

    # widget rotation (camera orientation)
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))
    roll = np.deg2rad(float(roll_deg))

    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]], dtype=np.float32)
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]], dtype=np.float32)

    R = Rz @ Rx @ Ry
    t = np.array([0, 0, 4.0], dtype=np.float32)  # move camera away

    def proj(pts):
        uv = _project_points(pts, K, R, t)
        return np.round(uv).astype(int)

    # cube corners in [-1,1]^3
    corners = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ], dtype=np.float32) * 0.9

    uv = proj(corners)

    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        cv2.line(canvas, tuple(uv[a]), tuple(uv[b]), line_color, 1, cv2.LINE_AA)

    # axes from origin
    origin = np.array([[0, 0, 0]], dtype=np.float32)
    axes = [
        (np.array([[0, 0, 0], [1.4, 0, 0]], dtype=np.float32), axis_colors[0]),
        (np.array([[0, 0, 0], [0, 1.4, 0]], dtype=np.float32), axis_colors[1]),
        (np.array([[0, 0, 0], [0, 0, 1.4]], dtype=np.float32), axis_colors[2]),
    ]
    for seg, col in axes:
        uva = proj(seg)
        cv2.arrowedLine(canvas, tuple(uva[0]), tuple(uva[1]), col, 2, tipLength=0.2)

    # light vector
    v = np.array(light_dir_world, dtype=np.float32).reshape(3)
    vn = v / (np.linalg.norm(v) + 1e-6)
    seg = np.array([[0, 0, 0], vn * 1.6], dtype=np.float32)
    uvl = proj(seg)
    cv2.arrowedLine(canvas, tuple(uvl[0]), tuple(uvl[1]), vec_color, 2, tipLength=0.2)

    return canvas


def overlay_widget(base_bgr, widget_bgr, anchor='br', margin=12):
    """Nakłada widget na obraz bazowy w rogu."""
    if base_bgr is None or widget_bgr is None:
        return base_bgr
    out = base_bgr
    H, W = out.shape[:2]
    h, w = widget_bgr.shape[:2]
    margin = int(max(0, margin))

    if anchor == 'br':
        x0 = W - w - margin
        y0 = H - h - margin
    elif anchor == 'tr':
        x0 = W - w - margin
        y0 = margin
    elif anchor == 'bl':
        x0 = margin
        y0 = H - h - margin
    else:
        x0 = margin
        y0 = margin

    x0 = int(np.clip(x0, 0, max(0, W - w)))
    y0 = int(np.clip(y0, 0, max(0, H - h)))

    roi = out[y0:y0 + h, x0:x0 + w]
    if roi.shape[:2] != widget_bgr.shape[:2]:
        return out

    # simple alpha blend
    alpha = 0.9
    blended = cv2.addWeighted(roi, 1 - alpha, widget_bgr, alpha, 0)
    out[y0:y0 + h, x0:x0 + w] = blended
    return out
