from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from step1_harris_keypoints import detect_harris_keypoints, to_grayscale

# function summary:
# 1) extract_patch_descriptors: builds normalized fixed-size patch descriptors around keypoints.
# 2) match_descriptors_ncc: matches descriptors with normalized correlation and mutual best filtering.
# 3) run_patch_size_sensitivity: evaluates multiple patch sizes and reports comparative metrics.
# 4) main: loads images, runs the sensitivity analysis, and saves a result plot.


def extract_patch_descriptors(
    gray: np.ndarray,
    keypoints_rc: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    # fixed-size patches describe local appearance around each keypoint.
    # descriptors are z-score normalized so brightness scale has less influence on matching.
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")

    half = patch_size // 2
    h, w = gray.shape

    descriptors: list[np.ndarray] = []
    valid_kps: list[list[int]] = []

    for r, c in keypoints_rc:
        if r - half < 0 or c - half < 0 or r + half >= h or c + half >= w:
            continue

        patch = gray[r - half : r + half + 1, c - half : c + half + 1].astype(np.float32)
        vec = patch.reshape(-1)

        mu = float(np.mean(vec))
        sigma = float(np.std(vec))
        if sigma < 1e-8:
            continue

        vec = (vec - mu) / sigma
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            continue

        vec = vec / norm
        descriptors.append(vec)
        valid_kps.append([int(r), int(c)])

    if not descriptors:
        return np.empty((0, patch_size * patch_size), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

    return np.vstack(descriptors), np.asarray(valid_kps, dtype=np.int32)


def match_descriptors_ncc(desc1: np.ndarray, desc2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # normalized correlation is dot product because descriptors are unit-normalized.
    # mutual-best matching removes many one-sided ambiguous matches.
    if desc1.size == 0 or desc2.size == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
        )

    corr = desc1 @ desc2.T
    best_j_for_i = np.argmax(corr, axis=1)
    best_i_for_j = np.argmax(corr, axis=0)

    i_indices = np.arange(desc1.shape[0], dtype=np.int32)
    mutual = best_i_for_j[best_j_for_i] == i_indices

    matched_i = i_indices[mutual]
    matched_j = best_j_for_i[mutual].astype(np.int32)
    matched_scores = corr[matched_i, matched_j].astype(np.float32)

    return matched_i, matched_j, matched_scores


def run_patch_size_sensitivity(
    gray1: np.ndarray,
    gray2: np.ndarray,
    kp1: np.ndarray,
    kp2: np.ndarray,
    patch_sizes: list[int],
    corr_threshold: float,
) -> list[dict[str, float]]:
    # this compares descriptor size impact using a consistent matching rule.
    # later, replace or complement this with your assignment final score after ransac/warping.
    rows: list[dict[str, float]] = []

    for size in patch_sizes:
        d1, vkp1 = extract_patch_descriptors(gray1, kp1, size)
        d2, vkp2 = extract_patch_descriptors(gray2, kp2, size)

        mi, mj, ms = match_descriptors_ncc(d1, d2)
        good = ms >= corr_threshold

        num_desc1 = int(d1.shape[0])
        num_desc2 = int(d2.shape[0])
        num_mutual = int(ms.size)
        num_good = int(np.sum(good))

        mean_corr = float(np.mean(ms)) if ms.size else float("nan")
        mean_good_corr = float(np.mean(ms[good])) if np.any(good) else float("nan")

        rows.append(
            {
                "patch_size": float(size),
                "desc1": float(num_desc1),
                "desc2": float(num_desc2),
                "mutual_matches": float(num_mutual),
                "good_matches": float(num_good),
                "mean_corr": mean_corr,
                "mean_good_corr": mean_good_corr,
            }
        )

    return rows


def main() -> None:
    # runs patch-size sensitivity and stores a compact visualization.
    parser = argparse.ArgumentParser(description="Step 2: patch descriptor sensitivity analysis")
    parser.add_argument("--img1", type=str, default="car1.jpg")
    parser.add_argument("--img2", type=str, default="car2.jpeg")
    parser.add_argument("--patch_sizes", type=str, default="7,11,15,21")
    parser.add_argument("--corr_threshold", type=float, default=0.75)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--k", type=float, default=0.05)
    parser.add_argument("--threshold_rel", type=float, default=0.02)
    parser.add_argument("--min_distance", type=int, default=8)
    parser.add_argument("--max_points", type=int, default=800)
    parser.add_argument("--out_plot", type=str, default="outputs/patch_size_sensitivity.png")
    args = parser.parse_args()

    patch_sizes = [int(x.strip()) for x in args.patch_sizes.split(",") if x.strip()]
    for p in patch_sizes:
        if p <= 1 or p % 2 == 0:
            raise ValueError("all patch sizes must be odd integers >= 3")

    img1 = io.imread(args.img1)
    img2 = io.imread(args.img2)
    gray1 = to_grayscale(img1)
    gray2 = to_grayscale(img2)

    kp1 = detect_harris_keypoints(gray1, args.sigma, args.k, args.threshold_rel, args.min_distance, args.max_points)
    kp2 = detect_harris_keypoints(gray2, args.sigma, args.k, args.threshold_rel, args.min_distance, args.max_points)

    rows = run_patch_size_sensitivity(gray1, gray2, kp1, kp2, patch_sizes, args.corr_threshold)

    print("patch sensitivity summary")
    print("size | desc1 | desc2 | mutual | good | mean_corr | mean_good_corr")
    for r in rows:
        print(
            f"{int(r['patch_size']):>4} | "
            f"{int(r['desc1']):>5} | "
            f"{int(r['desc2']):>5} | "
            f"{int(r['mutual_matches']):>6} | "
            f"{int(r['good_matches']):>4} | "
            f"{r['mean_corr']:.4f} | "
            f"{r['mean_good_corr']:.4f}"
        )

    out_plot = Path(args.out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    x = [int(r["patch_size"]) for r in rows]
    good = [r["good_matches"] for r in rows]
    corr = [r["mean_corr"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x, good, marker="o", color="tab:blue", label="good matches")
    ax1.set_xlabel("patch size")
    ax1.set_ylabel("good matches", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(x, corr, marker="s", color="tab:orange", label="mean correlation")
    ax2.set_ylabel("mean correlation", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle("patch size sensitivity (descriptor + ncc matching)")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)

    print(f"saved sensitivity plot: {out_plot}")


if __name__ == "__main__":
    main()
