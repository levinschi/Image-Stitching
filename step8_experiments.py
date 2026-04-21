from pathlib import Path

import cv2
import numpy as np
from skimage import io

from stitch_utils import (
    as_rgb_float, harris_sift_points_desc, pairs_to_xy, run_ransac,
)

# steps
# 1) run the full stitching pipeline on multiple image pairs
# 2) for each pair: extract harris+sift, ratio match, ransac, warp and blend
# 3) save blended panorama and print inlier stats per pair


PAIRS = [
    ("sign1.jpeg",  "sign2.jpeg",  "outputs/step8_sign.png"),
    ("build1.jpeg", "build2.jpeg", "outputs/step8_build.png"),
    ("car1.jpg",    "car2.jpeg",   "outputs/step8_car.png"),
]

SIGMA          = 1.5
K              = 0.05
THRESHOLD_REL  = 0.01
MIN_DISTANCE   = 5
MAX_POINTS     = 1000
KEYPOINT_SIZE  = 8.0
RATIO_TEST     = 0.75
TOPK           = 200
SAMPLE_SIZE    = 3
ITERATIONS     = 600
THR_RANSAC     = 5.0 #found after step5
SEED           = 42


def ratio_match(d1, d2):
    # keep only matches where the closest descriptor is clearly better than the second closest.
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < RATIO_TEST * n.distance]
    if not good:
        return np.empty((0, 2), dtype=int)
    good.sort(key=lambda m: m.distance)
    return np.array([(m.queryIdx, m.trainIdx) for m in good], dtype=int)


def unique_one_to_one(pairs, scores):
    # greedily keep the best match for each descriptor index on both sides.
    order = np.argsort(scores)
    used_i, used_j, kept = set(), set(), []
    for idx in order:
        i, j = int(pairs[idx, 0]), int(pairs[idx, 1])
        if i not in used_i and j not in used_j:
            used_i.add(i); used_j.add(j); kept.append([i, j])
    return np.asarray(kept, dtype=int) if kept else np.empty((0, 2), dtype=int)


def topk_unique(pairs):
    # ratio_match returns pairs sorted by distance. so take the first k and enforce uniqueness.
    top = pairs[:min(TOPK, len(pairs))]
    scores = np.arange(len(top), dtype=np.float32)  # rank order as score (lower = better)
    return unique_one_to_one(top, scores)


def to_rgb(image):
    # ensure image is 3 channel rgb regardless of input format.
    # different from the as_rgb_float in stitch_utils. this one does not convert to float.
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def transform_corners(H, w, h):
    corners = np.array([[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]], dtype=np.float32)
    return cv2.perspectiveTransform(corners, H.astype(np.float64))[0]


def feather_blend(img1, img2, H_2_to_1):
    # warp both images onto a common canvas and blend using distance-weighted feathering.
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    c1 = transform_corners(np.eye(3), w1, h1)
    c2 = transform_corners(H_2_to_1, w2, h2)
    all_c = np.vstack([c1, c2])
    min_x, min_y = np.floor(np.min(all_c, axis=0)).astype(int)
    max_x, max_y = np.ceil(np.max(all_c, axis=0)).astype(int)

    tx, ty = -min_x, -min_y
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    out_w, out_h = int(max_x - min_x + 1), int(max_y - min_y + 1)

    warp1  = cv2.warpPerspective(img1, T, (out_w, out_h), flags=cv2.INTER_LINEAR)
    warp2  = cv2.warpPerspective(img2, T @ H_2_to_1, (out_w, out_h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    m1 = cv2.warpPerspective(np.full((h1, w1), 255, dtype=np.uint8), T, (out_w, out_h), flags=cv2.INTER_NEAREST) > 0
    m2 = cv2.warpPerspective(np.full((h2, w2), 255, dtype=np.uint8), T @ H_2_to_1, (out_w, out_h), flags=cv2.INTER_NEAREST) > 0

    d1f = cv2.distanceTransform(m1.astype(np.uint8) * 255, cv2.DIST_L2, 3).astype(np.float32)
    d2f = cv2.distanceTransform(m2.astype(np.uint8) * 255, cv2.DIST_L2, 3).astype(np.float32)

    denom = d1f + d2f + 1e-6
    w1f, w2f = d1f / denom, d2f / denom
    w1f[m1 & ~m2], w2f[m1 & ~m2] = 1.0, 0.0
    w1f[~m1 & m2], w2f[~m1 & m2] = 0.0, 1.0
    w1f[~m1 & ~m2], w2f[~m1 & ~m2] = 0.0, 0.0

    blend = warp1.astype(np.float32) * w1f[..., None] + warp2.astype(np.float32) * w2f[..., None]
    return np.clip(blend, 0, 255).astype(np.uint8)


def stitch_pair(img1_path, img2_path, out_path):
    img1 = to_rgb(io.imread(img1_path))
    img2 = to_rgb(io.imread(img2_path))

    p1, d1 = harris_sift_points_desc(img1, SIGMA, K, THRESHOLD_REL, MIN_DISTANCE, MAX_POINTS, KEYPOINT_SIZE)
    p2, d2 = harris_sift_points_desc(img2, SIGMA, K, THRESHOLD_REL, MIN_DISTANCE, MAX_POINTS, KEYPOINT_SIZE)

    raw_pairs = ratio_match(d1, d2)
    pairs = topk_unique(raw_pairs)

    if len(pairs) < SAMPLE_SIZE:
        print(f"  not enough matches ({len(pairs)}), skipping")
        return

    src, dst = pairs_to_xy(p1, p2, pairs)
    result = run_ransac(src, dst, SAMPLE_SIZE, ITERATIONS, THR_RANSAC, np.random.default_rng(SEED))

    if result is None:
        print(f"  ransac failed, skipping")
        return

    n_inliers = result["inliers"]
    n_total   = len(pairs)
    print(f"  matches={n_total}  inliers={n_inliers}  ratio={n_inliers/n_total:.3f}")

    # extend the 2x3 affine to a 3x3 matrix and invert it to warp image2 onto image1's frame.
    A = result["M"]
    H_1_to_2 = np.vstack([A, [0, 0, 1]]).astype(np.float64)
    H_2_to_1 = np.linalg.inv(H_1_to_2)

    panorama = feather_blend(img1, img2, H_2_to_1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    io.imsave(out_path, panorama)
    print(f"  saved: {out_path}")


def main():
    for img1_path, img2_path, out_path in PAIRS:
        print(f"{img1_path} + {img2_path}")
        stitch_pair(img1_path, img2_path, out_path)


if __name__ == "__main__":
    main()
