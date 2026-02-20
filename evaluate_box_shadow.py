import argparse
import os
import csv

import cv2
import numpy as np

from src.shadow_detection import detect_shadow_physics
from src.shadow_direction import fuse_light_vectors


def angular_error_deg(a_deg, b_deg):
    """Minimalny blad katowy w stopniach (0..180)."""
    d = (float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0
    return abs(d)


def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    total = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        total += 1
    cap.release()
    return total


def evaluate_video(video_path,
                   start_angle_deg=35.0,
                   rotation_deg=360.0,
                   step=1,
                   max_frames=0,
                   save_csv_path=None):
    if not os.path.isfile(video_path):
        raise ValueError(f"Brak pliku: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Nie mozna otworzyc wideo: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        total_frames = count_frames(video_path)
        cap = cv2.VideoCapture(video_path)

    if total_frames <= 0:
        cap.release()
        raise ValueError("Nie mozna ustalic liczby klatek")

    rows = []
    errors = []
    confs = []

    idx = 0
    eval_count = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if idx % int(max(1, step)) != 0:
            idx += 1
            continue

        t = float(idx) / float(total_frames)
        gt_angle = (float(start_angle_deg) + float(rotation_deg) * t) % 360.0

        mask = detect_shadow_physics(frame)
        pred_angle, _vec, conf = fuse_light_vectors(frame, mask, debug=False)

        err = angular_error_deg(pred_angle, gt_angle)
        errors.append(float(err))
        confs.append(float(conf))

        rows.append({
            'frame_idx': int(idx),
            'gt_angle_deg': float(gt_angle),
            'pred_angle_deg': float(pred_angle),
            'error_deg': float(err),
            'confidence': float(conf),
        })

        eval_count += 1
        if int(max_frames) > 0 and eval_count >= int(max_frames):
            break

        idx += 1

    cap.release()

    if not rows:
        raise ValueError("Brak danych do ewaluacji (0 klatek)")

    errors = np.array(errors, dtype=np.float32)
    confs = np.array(confs, dtype=np.float32)

    def mae_for_thr(thr):
        m = confs >= float(thr)
        if m.sum() == 0:
            return None, 0.0
        return float(errors[m].mean()), float(m.sum()) / float(len(errors))

    mae_total = float(errors.mean())
    rmse_total = float(np.sqrt(np.mean(errors ** 2)))
    mae_06, cov_06 = mae_for_thr(0.6)
    mae_05, cov_05 = mae_for_thr(0.5)
    mae_07, cov_07 = mae_for_thr(0.7)

    if save_csv_path:
        with open(save_csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    result = {
        'video': video_path,
        'frames_total': int(total_frames),
        'frames_eval': int(len(rows)),
        'start_angle_deg': float(start_angle_deg),
        'rotation_deg': float(rotation_deg),
        'step': int(step),
        'mae_total': float(mae_total),
        'rmse_total': float(rmse_total),
        'mae_conf>0.5': None if mae_05 is None else float(mae_05),
        'mae_conf>0.6': None if mae_06 is None else float(mae_06),
        'mae_conf>0.7': None if mae_07 is None else float(mae_07),
        'coverage_conf>0.5': float(cov_05),
        'coverage_conf>0.6': float(cov_06),
        'coverage_conf>0.7': float(cov_07),
        'mean_conf': float(np.mean(confs)),
        'median_conf': float(np.median(confs)),
        'min_conf': float(np.min(confs)),
        'max_conf': float(np.max(confs)),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate shadow direction on box-shadow.mp4")
    parser.add_argument('--video', default=os.path.join('data', 'box-shadow.mp4'))
    parser.add_argument('--start-angle', type=float, default=35.0)
    parser.add_argument('--rotation-deg', type=float, default=360.0)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--save-csv', default='')
    args = parser.parse_args()

    result = evaluate_video(
        video_path=args.video,
        start_angle_deg=args.start_angle,
        rotation_deg=args.rotation_deg,
        step=args.step,
        max_frames=args.max_frames,
        save_csv_path=args.save_csv if args.save_csv else None,
    )

    # raport
    print("video:", result['video'])
    print("frames_total:", result['frames_total'])
    print("frames_eval:", result['frames_eval'])
    print("start_angle_deg:", f"{result['start_angle_deg']:.1f}")
    print("rotation_deg:", f"{result['rotation_deg']:.1f}")
    print("step:", result['step'])
    print("MAE_total:", f"{result['mae_total']:.2f}")
    print("RMSE_total:", f"{result['rmse_total']:.2f}")
    print("MAE_conf>0.5:", "n/a" if result['mae_conf>0.5'] is None else f"{result['mae_conf>0.5']:.2f}")
    print("MAE_conf>0.6:", "n/a" if result['mae_conf>0.6'] is None else f"{result['mae_conf>0.6']:.2f}")
    print("MAE_conf>0.7:", "n/a" if result['mae_conf>0.7'] is None else f"{result['mae_conf>0.7']:.2f}")
    print("coverage_conf>0.5:", f"{result['coverage_conf>0.5']:.2f}")
    print("coverage_conf>0.6:", f"{result['coverage_conf>0.6']:.2f}")
    print("coverage_conf>0.7:", f"{result['coverage_conf>0.7']:.2f}")
    print("mean_conf:", f"{result['mean_conf']:.2f}")
    print("median_conf:", f"{result['median_conf']:.2f}")
    print("min_conf:", f"{result['min_conf']:.2f}")
    print("max_conf:", f"{result['max_conf']:.2f}")


if __name__ == '__main__':
    main()

