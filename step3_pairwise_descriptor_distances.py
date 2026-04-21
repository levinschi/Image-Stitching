from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from stitch_utils import harris_sift_points_desc

# steps
# 1) extract sift descriptors on harris keypoints for both images
# 2) l2-normalize every descriptor so dot product equals cosine similarity
# 3) compute full pairwise normalized correlation matrix (n1 x n2) via matrix multiply
# 4) derive euclidean distance matrix from correlation using ||u-v|| = sqrt(2 - 2*corr)
# 5) save both matrices as .npy and plot score distributions as histograms


def normalize_desc(desc):
    # divide each descriptor by its own l2 norm so all vectors have unit length.
    # after this, dot product between two descriptors equals their cosine similarity.
    if len(desc) == 0:
        return desc
    return (desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def save_hist(corr, euc, out_path, bins):
    # plot the distribution of both metrics across all descriptor pairs.
    # the dashed line shows the median, which helps compare the two distributions.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    c, e = corr.reshape(-1), euc.reshape(-1)
    ax[0].hist(c, bins=bins, color="tab:blue", alpha=0.8)
    ax[0].axvline(float(np.median(c)), color="k", ls="--", lw=1)
    ax[0].set_title("normalized correlation")
    ax[0].set_xlabel("score")
    ax[0].set_ylabel("count")
    ax[1].hist(e, bins=bins, color="tab:orange", alpha=0.8)
    ax[1].axvline(float(np.median(e)), color="k", ls="--", lw=1)
    ax[1].set_title("normalized euclidean distance")
    ax[1].set_xlabel("distance")
    ax[1].set_ylabel("count")
    fig.suptitle("step 3: pairwise descriptor score distributions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="step 3: full pairwise descriptor distances")
    p.add_argument("--img1", default="mnt1.JPG")
    p.add_argument("--img2", default="mnt2.JPG")
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--k", type=float, default=0.05)
    p.add_argument("--threshold_rel", type=float, default=0.01)
    p.add_argument("--min_distance", type=int, default=5)
    p.add_argument("--max_points", type=int, default=1000)
    p.add_argument("--sift_keypoint_size", type=float, default=8.0)
    p.add_argument("--out_corr_npy", default="outputs/step3_norm_correlation.npy")
    p.add_argument("--out_euc_npy", default="outputs/step3_norm_euclidean.npy")
    p.add_argument("--out_hist_png", default="outputs/step3_pairwise_histograms.png")
    p.add_argument("--hist_bins", type=int, default=50)
    a = p.parse_args()

    img1, img2 = io.imread(a.img1), io.imread(a.img2)

    # reuse harris keypoints as fixed frames, compute sift descriptors at those locations
    _, d1 = harris_sift_points_desc(img1, a.sigma, a.k, a.threshold_rel,
                                     a.min_distance, a.max_points, a.sift_keypoint_size)
    _, d2 = harris_sift_points_desc(img2, a.sigma, a.k, a.threshold_rel,
                                     a.min_distance, a.max_points, a.sift_keypoint_size)

    n1, n2 = normalize_desc(d1), normalize_desc(d2)

    # normalized correlation: dot product of unit vectors = cosine similarity
    corr = np.clip(n1 @ n2.T, -1.0, 1.0).astype(np.float32)
    # euclidean distance derived from correlation: ||u-v|| = sqrt(2 - 2*(u.v))
    # avoids computing the full distance matrix a second time
    euc = np.sqrt(np.maximum(2.0 - 2.0 * corr, 0.0)).astype(np.float32)

    for f in [a.out_corr_npy, a.out_euc_npy, a.out_hist_png]:
        Path(f).parent.mkdir(parents=True, exist_ok=True)

    np.save(a.out_corr_npy, corr)
    np.save(a.out_euc_npy, euc)
    save_hist(corr, euc, a.out_hist_png, bins=a.hist_bins)

    print(f"img1 descriptors: {len(d1)}")
    print(f"img2 descriptors: {len(d2)}")
    print(f"pairwise matrix shape: {corr.shape}")
    print(f"correlation  min/max/mean: {corr.min():.4f} / {corr.max():.4f} / {corr.mean():.4f}")
    print(f"euclidean    min/max/mean: {euc.min():.4f} / {euc.max():.4f} / {euc.mean():.4f}")
    print(f"correlation  median: {float(np.median(corr)):.4f}")
    print(f"euclidean    median: {float(np.median(euc)):.4f}")
    print(f"saved: {a.out_corr_npy}")
    print(f"saved: {a.out_euc_npy}")
    print(f"saved: {a.out_hist_png}")


if __name__ == "__main__":
    main()
