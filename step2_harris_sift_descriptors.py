from pathlib import Path
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from stitch_utils import as_rgb_float, harris_sift_points_desc

# steps
# 1) detect harris points in both images
# 2) compute sift descriptors on those points
# 3) match descriptors with ratio test
# 4) save keypoint and match visualizations


def ratio_match(desc1, desc2, ratio_test):
    # ratio test reject ambiguous nearest-neighbor matches.
    # knn with k=2, keep first if distance ratio is small enough.
    if len(desc1) == 0 or len(desc2) < 2:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=np.float32)

    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc1, desc2, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio_test * n.distance]
    good.sort(key=lambda m: m.distance)
    return np.array([(m.queryIdx, m.trainIdx) for m in good], dtype=int), np.array([m.distance for m in good], dtype=np.float32)


def save_keypoints(img1, p1, img2, p2, out_path):
    # visual plot helps check if harris->sift point extraction looks normal.
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for a, img, p, t in [(ax[0], img1, p1, "img1"), (ax[1], img2, p2, "img2")]:
        a.imshow(img)
        a.scatter(p[:, 1], p[:, 0], s=10, c="lime")
        a.set_title(f"{t} harris->sift | keypoints={len(p)}")
        a.axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def save_matches(img1, p1, img2, p2, matches, out_path, max_lines):
    # correspondence plot let us see if matching is mostly geometrically good.
   
    v1, v2 = as_rgb_float(img1), as_rgb_float(img2)
    h1, w1 = v1.shape[:2]
    h2, w2 = v2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.float32)
    canvas[:h1, :w1], canvas[:h2, w1 : w1 + w2] = v1, v2

    fig, ax = plt.subplots(1, 1, figsize=(14, 6)); ax.imshow(canvas)
    for i, j in matches[:max_lines]:
        y1, x1 = p1[i]; y2, x2 = p2[j]
        ax.plot([x1, x2 + w1], [y1, y2], color="yellow", linewidth=0.7, alpha=0.6)
    ax.set_title(f"sift on harris points | matches={len(matches)}"); ax.axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="step 2: sift descriptors on harris points")
    p.add_argument("--img1", default="mnt1.JPG"); p.add_argument("--img2", default="mnt2.JPG")
    p.add_argument("--sigma", type=float, default=1.5); p.add_argument("--k", type=float, default=0.05)
    p.add_argument("--threshold_rel", type=float, default=0.01); p.add_argument("--min_distance", type=int, default=5)
    p.add_argument("--max_points", type=int, default=1000); p.add_argument("--sift_keypoint_size", type=float, default=8.0)
    p.add_argument("--ratio_test", type=float, default=0.75); p.add_argument("--max_draw_matches", type=int, default=150)
    p.add_argument("--out_keypoints", default="outputs/harris_keypoints_for_sift.png")
    p.add_argument("--out_matches", default="outputs/harris_sift_matches.png")
    a = p.parse_args()

    img1, img2 = io.imread(a.img1), io.imread(a.img2)
    p1, d1 = harris_sift_points_desc(img1, a.sigma, a.k, a.threshold_rel, a.min_distance, a.max_points, a.sift_keypoint_size)
    p2, d2 = harris_sift_points_desc(img2, a.sigma, a.k, a.threshold_rel, a.min_distance, a.max_points, a.sift_keypoint_size)
    matches, dist = ratio_match(d1, d2, a.ratio_test)

    Path(a.out_keypoints).parent.mkdir(parents=True, exist_ok=True)
    save_keypoints(img1, p1, img2, p2, a.out_keypoints)
    save_matches(img1, p1, img2, p2, matches, a.out_matches, a.max_draw_matches)

    print(f"saved keypoint figure: {a.out_keypoints}")
    print(f"saved match figure: {a.out_matches}")
    print(f"img1 sift descriptors on harris points: {len(d1)}")
    print(f"img2 sift descriptors on harris points: {len(d2)}")
    print(f"matches after ratio test: {len(matches)}")
    print(f"median sift match distance: {float(np.median(dist)):.4f}")


if __name__ == "__main__":
    main()
