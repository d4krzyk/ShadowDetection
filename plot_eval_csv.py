import argparse
import csv
import os
import sys

import numpy as np


def _load_csv_rows(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _to_float(val, default=np.nan):
    try:
        return float(val)
    except Exception:
        return float(default)


def _rolling_mean(x, k):
    if k <= 1:
        return x
    k = int(k)
    if k <= 1 or k > len(x):
        return x
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x, kernel, mode="same")


def _summarize(errors, confs):
    errors = np.array(errors, dtype=np.float32)
    confs = np.array(confs, dtype=np.float32)

    def mae_for_thr(thr):
        m = confs >= float(thr)
        if m.sum() == 0:
            return None, 0.0
        return float(np.mean(errors[m])), float(m.sum()) / float(len(errors))

    mae_total = float(np.mean(errors))
    rmse_total = float(np.sqrt(np.mean(errors ** 2)))
    mae_05, cov_05 = mae_for_thr(0.5)
    mae_06, cov_06 = mae_for_thr(0.6)
    mae_07, cov_07 = mae_for_thr(0.7)

    return {
        "mae_total": mae_total,
        "rmse_total": rmse_total,
        "mae_conf>0.5": mae_05,
        "mae_conf>0.6": mae_06,
        "mae_conf>0.7": mae_07,
        "coverage_conf>0.5": cov_05,
        "coverage_conf>0.6": cov_06,
        "coverage_conf>0.7": cov_07,
        "mean_conf": float(np.mean(confs)),
        "median_conf": float(np.median(confs)),
        "min_conf": float(np.min(confs)),
        "max_conf": float(np.max(confs)),
    }


def _pick_csv_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Wybierz CSV z ewaluacji",
        filetypes=[("CSV", "*.csv"), ("Wszystkie pliki", "*.*")],
    )
    root.destroy()
    return path


def plot_csv(csv_path, smooth=1, show=True, save_path=None):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"Brak matplotlib: {e}")

    rows = _load_csv_rows(csv_path)
    if not rows:
        raise ValueError("CSV jest pusty")

    frames = [int(_to_float(r.get("frame_idx", i))) for i, r in enumerate(rows)]
    gt = [_to_float(r.get("gt_angle_deg")) for r in rows]
    pred = [_to_float(r.get("pred_angle_deg")) for r in rows]
    err = [_to_float(r.get("error_deg")) for r in rows]
    conf = [_to_float(r.get("confidence")) for r in rows]

    err_s = _rolling_mean(np.array(err, dtype=np.float32), smooth)
    conf_s = _rolling_mean(np.array(conf, dtype=np.float32), smooth)

    stats = _summarize(err, conf)

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(os.path.basename(csv_path))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(frames, err, label="error_deg", color="#d62728", alpha=0.5)
    ax1.plot(frames, err_s, label=f"error_deg (smooth={smooth})", color="#d62728")
    ax1.set_title("Blad katowy")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Error (deg)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(frames, conf, label="confidence", color="#2ca02c", alpha=0.5)
    ax2.plot(frames, conf_s, label=f"confidence (smooth={smooth})", color="#2ca02c")
    ax2.set_title("Confidence")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Conf")
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(err, bins=30, color="#1f77b4", alpha=0.8)
    ax3.set_title("Histogram bledu")
    ax3.set_xlabel("Error (deg)")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(conf, err, s=12, alpha=0.6, color="#9467bd")
    ax4.set_title("Error vs Confidence")
    ax4.set_xlabel("Confidence")
    ax4.set_ylabel("Error (deg)")
    ax4.grid(True, alpha=0.3)

    txt = (
        f"MAE={stats['mae_total']:.2f} | RMSE={stats['rmse_total']:.2f}\n"
        f"MAE>0.5={_fmt(stats['mae_conf>0.5'])}  cov>0.5={stats['coverage_conf>0.5']:.2f}\n"
        f"MAE>0.6={_fmt(stats['mae_conf>0.6'])}  cov>0.6={stats['coverage_conf>0.6']:.2f}\n"
        f"MAE>0.7={_fmt(stats['mae_conf>0.7'])}  cov>0.7={stats['coverage_conf>0.7']:.2f}\n"
        f"mean_conf={stats['mean_conf']:.2f}  median_conf={stats['median_conf']:.2f}"
    )
    fig.text(0.02, 0.02, txt, fontsize=10)

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


def _fmt(v):
    if v is None:
        return "n/a"
    return f"{float(v):.2f}"


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation CSV (errors/confidence)")
    parser.add_argument("--csv", default="")
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--save", default="")
    args = parser.parse_args()

    csv_path = args.csv.strip() if args.csv else ""
    if not csv_path:
        csv_path = _pick_csv_dialog()

    if not csv_path:
        print("Brak pliku CSV", file=sys.stderr)
        sys.exit(2)

    plot_csv(csv_path, smooth=int(args.smooth), show=True, save_path=(args.save or None))


if __name__ == "__main__":
    main()

