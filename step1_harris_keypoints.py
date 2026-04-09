from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io, img_as_float
from skimage.feature import corner_harris, corner_peaks

# function summary:
# 1) to_grayscale: normalizes input images into a consistent grayscale format for stable corner detection.
# 2) detect_harris_keypoints: computes harris response and selects local peaks as candidate keypoints.
# 3) plot_keypoints: visualizes detected keypoint locations over each image for quality inspection.
# 4) main: parses parameters, runs the step-1 pipeline for two images, and saves a comparison figure.


def to_grayscale(image: np.ndarray) -> np.ndarray:
    # harris uses intensity gradients, so a single-channel image improves consistency.
    # keep existing 2d inputs, otherwise convert rgb to grayscale using skimage utilities.
    if image.ndim == 2:
        return img_as_float(image)
    return color.rgb2gray(image)


def detect_harris_keypoints(
    gray: np.ndarray,
    sigma: float,
    k: float,
    threshold_rel: float,
    min_distance: int,
    max_points: int,
) -> np.ndarray:
    # the raw harris response is dense; we need distinct, repeatable keypoints for later matching.
    # first compute corner strength, then apply non-maximum suppression + thresholding via corner_peaks.
    
    response = corner_harris(gray, method="k", k=k, sigma=sigma)
    points_rc = corner_peaks(
        response,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        num_peaks=max_points,
    )
    # returned coordinates are (row, col), which matches numpy image indexing.
    return points_rc


def plot_keypoints(image: np.ndarray, keypoints_rc: np.ndarray, title: str, ax: plt.Axes) -> None:
    # visual inspection helps quickly verify if parameters find too few, too many, or bad corners.
    # plot image first, then overlay (col, row) as (x, y) points, and display keypoint count in title.
    ax.imshow(image)
    if keypoints_rc.size > 0:
        ax.scatter(keypoints_rc[:, 1], keypoints_rc[:, 0], s=10, c="lime", marker="o")
    ax.set_title(f"{title} | keypoints={len(keypoints_rc)}")
    ax.axis("off")


def main() -> None:
    # central entrypoint makes experiments reproducible and easy to tune via command-line parameters.
    # load images, detect keypoints in both, create a side-by-side figure, save results, print counts.
    parser = argparse.ArgumentParser(description="Step 1: Harris keypoint detection visualization")
    parser.add_argument("--img1", type=str, default="car1.jpg")
    parser.add_argument("--img2", type=str, default="car2.jpeg")
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--k", type=float, default=0.05)
    parser.add_argument("--threshold_rel", type=float, default=0.01)
    parser.add_argument("--min_distance", type=int, default=5)
    parser.add_argument("--max_points", type=int, default=1000)
    parser.add_argument("--out", type=str, default="outputs/harris_keypoints_car_pair.png")
    args = parser.parse_args()

    img1 = io.imread(args.img1)
    img2 = io.imread(args.img2)

    # using the same preprocessing for both images makes detection comparable.
    gray1 = to_grayscale(img1)
    gray2 = to_grayscale(img2)

    # identical detector settings across both images support a fair step-1 baseline.
    kp1 = detect_harris_keypoints(
        gray1,
        sigma=args.sigma,
        k=args.k,
        threshold_rel=args.threshold_rel,
        min_distance=args.min_distance,
        max_points=args.max_points,
    )
    kp2 = detect_harris_keypoints(
        gray2,
        sigma=args.sigma,
        k=args.k,
        threshold_rel=args.threshold_rel,
        min_distance=args.min_distance,
        max_points=args.max_points,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # side-by-side layout helps directly compare corner distribution between the two views.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_keypoints(img1, kp1, "car1", axes[0])
    plot_keypoints(img2, kp2, "car2", axes[1])
    fig.suptitle("Harris keypoints")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved keypoint visualization: {out_path}")
    print(f"car1 keypoints: {len(kp1)}")
    print(f"car2 keypoints: {len(kp2)}")


if __name__ == "__main__":
    main()
