from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from stitch_utils import as_rgb_float, harris_sift_points_desc, pairs_to_xy, parse_list, run_ransac

# steps
# 1) load matched pairs from step 4 and rebuild matched point coordinates
# 2) sweep all combinations of iteration counts and inlier thresholds
# 3) keep the model with the most inliers, tie-break by lowest average squared residual
# 4) report inliers, outliers, inlier ratio, inlier/outlier ratio, avg squared residual
# 5) save affine matrix, inlier pair indices, and inlier correspondence visualization


def draw_inliers(img1, p1, img2, p2, pairs, mask, out_path, max_lines=200):
    # show only the geometrically consistent matches (inliers) on a side-by-side canvas
    v1, v2 = as_rgb_float(img1), as_rgb_float(img2)
    h1, w1 = v1.shape[:2]
    h2, w2 = v2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.float32)
    canvas[:h1, :w1] = v1
    canvas[:h2, w1: w1 + w2] = v2

    inlier_pairs = pairs[mask]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(canvas)
    shown = inlier_pairs[:max_lines]
    for i, j in shown:
        y1, x1 = p1[i]
        y2, x2 = p2[j]
        ax.plot([x1, x2 + w1], [y1, y2], color="lime", lw=0.7, alpha=0.7)
    if len(shown):
        ax.scatter(p1[shown[:, 0], 1], p1[shown[:, 0], 0], s=10, c="lime", zorder=3)
        ax.scatter(p2[shown[:, 1], 1] + w1, p2[shown[:, 1], 0], s=10, c="lime", zorder=3)
    ax.set_title(f"affine ransac inliers | inliers={mask.sum()} / {len(pairs)}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="step 5: affine transformation with custom ransac")
    p.add_argument("--img1", default="mnt1.JPG")
    p.add_argument("--img2", default="mnt2.JPG")
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--k", type=float, default=0.05)
    p.add_argument("--threshold_rel", type=float, default=0.01)
    p.add_argument("--min_distance", type=int, default=5)
    p.add_argument("--max_points", type=int, default=1000)
    p.add_argument("--sift_keypoint_size", type=float, default=8.0)
    p.add_argument("--pairs_npy", default="outputs/step4_corr_topk_pairs.npy")
    p.add_argument("--sample_size", type=int, default=3,
                   help="number of point pairs per ransac sample (3 is minimal for affine)")
    p.add_argument("--iterations", default="300,600,1000",
                   help="comma-separated ransac iteration counts to try")
    p.add_argument("--thresholds", default="2.0,3.0,4.0,5.0",
                   help="comma-separated inlier distance thresholds to try (pixels)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_affine", default="outputs/step5_ransac_affine.npy")
    p.add_argument("--out_inlier_pairs", default="outputs/step5_ransac_inlier_pairs.npy")
    p.add_argument("--out_inliers_png", default="outputs/step5_ransac_inliers.png")
    a = p.parse_args()

    iter_values = parse_list(a.iterations, int)
    thr_values = parse_list(a.thresholds, float)

    pairs = np.load(a.pairs_npy)

    img1 = io.imread(a.img1)
    img2 = io.imread(a.img2)
    p1, _ = harris_sift_points_desc(img1, a.sigma, a.k, a.threshold_rel,
                                     a.min_distance, a.max_points, a.sift_keypoint_size)
    p2, _ = harris_sift_points_desc(img2, a.sigma, a.k, a.threshold_rel,
                                     a.min_distance, a.max_points, a.sift_keypoint_size)
    src, dst = pairs_to_xy(p1, p2, pairs)

    # sweep all combinations of iterations x threshold and keep the model with the most inliers.
    # tie-break by lower average squared residual to prefer the more accurate model.
    best = None
    best_iters = None
    best_thr = None

    for iters in iter_values:
        for thr in thr_values:
            # unique seed per combination so each sweep is independently random
            rng = np.random.default_rng(a.seed + hash((iters, thr)) % (2**31))
            result = run_ransac(src, dst, a.sample_size, iters, thr, rng)
            if result is None:
                continue
            if (best is None
                    or result["inliers"] > best["inliers"]
                    or (result["inliers"] == best["inliers"] and result["avg"] < best["avg"])):
                best = result
                best_iters = iters
                best_thr = thr

    if best is None:
        raise RuntimeError("ransac failed for all parameter combinations")

    n_inliers = int(best["inliers"])
    n_outliers = len(pairs) - n_inliers
    inlier_ratio = n_inliers / len(pairs)
    # inlier/outlier ratio measures how clean the match set is
    inlier_outlier_ratio = n_inliers / n_outliers if n_outliers > 0 else float("inf")
    avg_sq_residual = float(best["avg"])

    for f in [a.out_affine, a.out_inlier_pairs, a.out_inliers_png]:
        Path(f).parent.mkdir(parents=True, exist_ok=True)

    np.save(a.out_affine, best["M"])
    np.save(a.out_inlier_pairs, pairs[best["mask"]])
    draw_inliers(img1, p1, img2, p2, pairs, best["mask"], a.out_inliers_png)

    print(f"total matched pairs:     {len(pairs)}")
    print(f"best settings:           iterations={best_iters}, threshold={best_thr}")
    print(f"inliers:                 {n_inliers}")
    print(f"outliers:                {n_outliers}")
    print(f"inlier ratio:            {inlier_ratio:.3f}")
    print(f"inlier/outlier ratio:    {inlier_outlier_ratio:.3f}")
    print(f"avg squared residual:    {avg_sq_residual:.6f}")
    print(f"affine matrix:{best['M']}")
    print(f"saved: {a.out_affine}")
    print(f"saved: {a.out_inlier_pairs}")
    print(f"saved: {a.out_inliers_png}")


if __name__ == "__main__":
    main()
