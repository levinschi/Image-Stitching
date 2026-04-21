from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from stitch_utils import harris_sift_points_desc, pairs_to_xy, parse_list, residuals_sq, run_ransac

# steps
# 1) load pairs from step 4 and rebuild matched point coordinates
# 2) define accuracy as mean euclidean inlier error
# 3) sweep sample size, iterations, and inlier threshold
# 4) save 3-panel sensitivity plot


def build_points(a):
    # rebuild the matched x/y coordinates from the saved pair indices
    img1, img2 = io.imread(a.img1), io.imread(a.img2)
    p1, _ = harris_sift_points_desc(img1, a.sigma, a.k, a.threshold_rel,
                                     a.min_distance, a.max_points, a.sift_keypoint_size)
    p2, _ = harris_sift_points_desc(img2, a.sigma, a.k, a.threshold_rel,
                                     a.min_distance, a.max_points, a.sift_keypoint_size)
    pairs = np.load(a.pairs_npy)
    return pairs_to_xy(p1, p2, pairs)


def accuracy(src, dst, sample_size, iterations, threshold, seed):
    # run ransac and return mean euclidean error over inliers (lower is better)
    r = run_ransac(src, dst, sample_size, iterations, threshold, np.random.default_rng(seed))
    if r is None:
        return np.nan
    e = np.sqrt(residuals_sq(r["M"], src, dst))
    return float(np.mean(e[r["mask"]]))


def main():
    p = argparse.ArgumentParser(description="step 7: euclidean accuracy sensitivity")
    p.add_argument("--img1", default="mnt1.JPG")
    p.add_argument("--img2", default="mnt2.JPG")
    p.add_argument("--pairs_npy", default="outputs/step4_corr_topk_pairs.npy")
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--k", type=float, default=0.05)
    p.add_argument("--threshold_rel", type=float, default=0.01)
    p.add_argument("--min_distance", type=int, default=5)
    p.add_argument("--max_points", type=int, default=1000)
    p.add_argument("--sift_keypoint_size", type=float, default=8.0)
    p.add_argument("--base_sample_size", type=int, default=3)
    p.add_argument("--base_iterations", type=int, default=600)
    p.add_argument("--base_threshold", type=float, default=4.0)
    p.add_argument("--sample_sizes", default="3,4,5")
    p.add_argument("--iter_values", default="100,300,600,1000")
    p.add_argument("--threshold_values", default="2.0,3.0,4.0,5.0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_plot", default="outputs/step7_accuracy_sensitivity.png")
    a = p.parse_args()

    src, dst = build_points(a)
    ss = parse_list(a.sample_sizes, int)
    it = parse_list(a.iter_values, int)
    th = parse_list(a.threshold_values, float)

    # each sweep varies one parameter at a time while keeping the others fixed.
    # seeds are offset per sweep so each run gets a different but reproducible random state.
    ss_scores = [accuracy(src, dst, v, a.base_iterations, a.base_threshold, a.seed + i)
                 for i, v in enumerate(ss)]
    it_scores = [accuracy(src, dst, a.base_sample_size, v, a.base_threshold, a.seed + 100 + i)
                 for i, v in enumerate(it)]
    th_scores = [accuracy(src, dst, a.base_sample_size, a.base_iterations, v, a.seed + 200 + i)
                 for i, v in enumerate(th)]

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    ax[0].plot(ss, ss_scores, marker="o"); ax[0].set_title("sample size"); ax[0].set_xlabel("sample size")
    ax[1].plot(it, it_scores, marker="o"); ax[1].set_title("iterations");  ax[1].set_xlabel("iterations")
    ax[2].plot(th, th_scores, marker="o"); ax[2].set_title("inlier threshold"); ax[2].set_xlabel("threshold")
    for axy in ax:
        axy.set_ylabel("mean inlier euclidean error")
        axy.grid(alpha=0.3)
    fig.suptitle("step 7: sensitivity using euclidean accuracy score")
    fig.tight_layout()

    out = Path(a.out_plot)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
