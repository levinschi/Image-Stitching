from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.measure import ransac
from skimage.transform import AffineTransform

from stitch_utils import as_rgb_float, detect_harris_points, parse_list, patch_descriptors, to_gray_float

# steps
# 1) detect harris keypoints in both images
# 2) extract fixed-size normalized patch descriptors around each keypoint
# 3) match descriptors with ratio test + mutual nearest neighbor
# 4) score each patch size by ransac inlier count (sensitivity analysis)
# 5) save keypoint figure and best-match figure


def match_descriptors(desc1, desc2, ratio_threshold):
    # for each descriptor in desc1 find its two nearest neighbors in desc2.
    # keep the match only if the closest distance is much smaller than the second
    # closest (lowe ratio test) and the match is also mutual (both sides agree).
    if len(desc1) == 0 or len(desc2) < 2:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=np.float32)

    dist = (
        np.sum(desc1 ** 2, axis=1, keepdims=True)
        + np.sum(desc2 ** 2, axis=1)
        - 2.0 * (desc1 @ desc2.T)
    )
    dist = np.maximum(dist, 0.0)

    # two nearest neighbors in desc2 for each row in desc1
    nn2_idx = np.argpartition(dist, kth=1, axis=1)[:, :2]
    nn2_d = dist[np.arange(len(desc1))[:, None], nn2_idx]
    order = np.argsort(nn2_d, axis=1)
    first = nn2_idx[np.arange(len(desc1)), order[:, 0]]
    second = nn2_idx[np.arange(len(desc1)), order[:, 1]]

    d_first = dist[np.arange(len(desc1)), first]
    d_second = dist[np.arange(len(desc1)), second] + 1e-12

    ratio_ok = (d_first / d_second) < ratio_threshold
    mutual_ok = np.argmin(dist, axis=0)[first] == np.arange(len(desc1))

    keep = np.where(ratio_ok & mutual_ok)[0]
    if len(keep) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=np.float32)

    pairs = np.column_stack((keep, first[keep])).astype(int)
    dists = np.sqrt(d_first[keep]).astype(np.float32)
    order = np.argsort(dists)
    return pairs[order], dists[order]


def count_inliers(p1, p2, matches, threshold):
    # estimate an affine transform on the matched points with ransac and count
    # how many matches fall within the reprojection threshold (inliers).
    # this is only used here to score patch sizes, not as the main pipeline ransac.
    if len(matches) < 3:
        return 0
    src = p1[matches[:, 0]][:, ::-1].astype(np.float64)
    dst = p2[matches[:, 1]][:, ::-1].astype(np.float64)
    try:
        _, inliers = ransac((src, dst), AffineTransform,
                            min_samples=3, residual_threshold=threshold, max_trials=500)
    except Exception:
        return 0
    return int(np.sum(inliers)) if inliers is not None else 0


def save_keypoints(img1, kp1, img2, kp2, out_path):
    # show detected corners as green dots on both images side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, img, kp, title in [(axes[0], img1, kp1, "img1"), (axes[1], img2, kp2, "img2")]:
        ax.imshow(img)
        if len(kp) > 0:
            ax.scatter(kp[:, 1], kp[:, 0], s=10, c="lime")
        ax.set_title(f"{title} | keypoints={len(kp)}")
        ax.axis("off")
    fig.suptitle("harris keypoints")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_matches(img1, p1, img2, p2, matches, out_path, title, max_lines=120):
    # draw correspondence lines between the two images on a single canvas
    v1, v2 = as_rgb_float(img1), as_rgb_float(img2)
    h1, w1 = v1.shape[:2]
    h2, w2 = v2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.float32)
    canvas[:h1, :w1] = v1
    canvas[:h2, w1: w1 + w2] = v2

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(canvas)
    shown = matches[:max_lines]
    for i, j in shown:
        y1, x1 = p1[i]; y2, x2 = p2[j]
        ax.plot([x1, x2 + w1], [y1, y2], color="yellow", lw=0.7, alpha=0.6)
    if len(shown):
        ax.scatter(p1[shown[:, 0], 1], p1[shown[:, 0], 0], s=8, c="lime")
        ax.scatter(p2[shown[:, 1], 1] + w1, p2[shown[:, 1], 0], s=8, c="lime")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="step 1: harris keypoints + patch descriptor sensitivity")
    #here we set the pair of image
    parser.add_argument("--img1", default="mnt1.JPG")
    parser.add_argument("--img2", default="mnt2.JPG")
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--k", type=float, default=0.05)
    parser.add_argument("--threshold_rel", type=float, default=0.01)
    parser.add_argument("--min_distance", type=int, default=5)
    parser.add_argument("--max_points", type=int, default=1000)
    parser.add_argument("--patch_sizes", default="7,11,15,21")
    parser.add_argument("--ratio_threshold", type=float, default=0.8)
    parser.add_argument("--ransac_threshold", type=float, default=4.0)
    #output
    parser.add_argument("--out_keypoints", default="outputs/harris_keypoints_mnt_pair.png")
    parser.add_argument("--out_matches", default="outputs/harris_patch_matches_best.png")
    args = parser.parse_args()

    patch_sizes = [s for s in parse_list(args.patch_sizes, int) if s > 1 and s % 2 == 1]

    img1 = io.imread(args.img1)
    img2 = io.imread(args.img2)
    gray1 = to_gray_float(img1)
    gray2 = to_gray_float(img2)

    kp1 = detect_harris_points(gray1, args.sigma, args.k, args.threshold_rel,
                                args.min_distance, args.max_points)
    kp2 = detect_harris_points(gray2, args.sigma, args.k, args.threshold_rel,
                                args.min_distance, args.max_points)

    Path(args.out_keypoints).parent.mkdir(parents=True, exist_ok=True)
    save_keypoints(img1, kp1, img2, kp2, args.out_keypoints)

    # sensitivity analysis: try each patch size and score by ransac inlier count
    rows = []
    for size in patch_sizes:
        d1, p1 = patch_descriptors(gray1, kp1, size)
        d2, p2 = patch_descriptors(gray2, kp2, size)
        matches, dists = match_descriptors(d1, d2, args.ratio_threshold)
        inliers = count_inliers(p1, p2, matches, args.ransac_threshold)
        ratio = inliers / len(matches) if len(matches) else 0.0
        median_d = float(np.median(dists)) if len(dists) else float("inf")
        rows.append({"size": size, "n1": len(d1), "n2": len(d2),
                     "matches": len(matches), "inliers": inliers,
                     "ratio": ratio, "median_d": median_d,
                     "p1": p1, "p2": p2, "m": matches})

    # pick the patch size with the most inliers, tie-break by inlier ratio then match count.
    best = max(rows, key=lambda r: (r["inliers"], r["ratio"], r["matches"]))
    save_matches(img1, best["p1"], img2, best["p2"], best["m"], args.out_matches,
                 f"best patch={best['size']} | matches={best['matches']} | inliers={best['inliers']}")

    print("patch-size sensitivity analysis")
    print("patch | desc1 | desc2 | matches | inliers | inlier_ratio | median_distance")
    print("-" * 76)
    for r in rows:
        med = f"{r['median_d']:.4f}" if np.isfinite(r["median_d"]) else "n/a"
        print(f"{r['size']:>5} | {r['n1']:>5} | {r['n2']:>5} | "
              f"{r['matches']:>7} | {r['inliers']:>7} | {r['ratio']:>12.3f} | {med:>15}")

    print(f"\nbest patch size: {best['size']} "
          f"(inliers={best['inliers']}, ratio={best['ratio']:.3f})")
    print(f"saved keypoints: {args.out_keypoints}")
    print(f"saved matches:   {args.out_matches}")
    print(f"img1 keypoints: {len(kp1)}  img2 keypoints: {len(kp2)}")


if __name__ == "__main__":
    main()
