import numpy as np
from skimage import color, img_as_float
from skimage.feature import corner_harris, corner_peaks
import cv2

def parse_list(text, cast):
    # split a comma-separated string and cast each token to the given type.
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def to_gray_float(image):
    # harris runs on a single float channel for stable gradient computation.
    if image.ndim == 2:
        return img_as_float(image)
    return color.rgb2gray(image)


def to_gray_u8(image):
    # opencv sift expects a uint8 image.
    gray = to_gray_float(image)
    return np.clip(gray * 255.0, 0, 255).astype(np.uint8)


def detect_harris_points(gray, sigma, k, threshold_rel, min_distance, max_points):
    # compute harris corner response and return the strongest local maxima.
    resp = corner_harris(gray, method="k", k=k, sigma=sigma)
    return corner_peaks(
        resp,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        num_peaks=max_points,
    )


def patch_descriptors(gray, points_rc, patch_size):
    # extract a flat normalized patch around each keypoint.
    # keep only points where the full patch fits inside the image,
    # then mean-center and l2-normalize each patch so it can be
    # compared with dot product (same as normalized correlation).
    half = patch_size // 2
    h, w = gray.shape
    ok = (
        (points_rc[:, 0] - half >= 0) & (points_rc[:, 0] + half < h)
        & (points_rc[:, 1] - half >= 0) & (points_rc[:, 1] + half < w)
    )
    pts = points_rc[ok]
    if len(pts) == 0:
        return np.empty((0, patch_size * patch_size), dtype=np.float32), pts

    desc = np.empty((len(pts), patch_size * patch_size), dtype=np.float32)
    for i, (r, c) in enumerate(pts):
        p = gray[r - half: r + half + 1, c - half: c + half + 1].astype(np.float32)
        p -= p.mean()
        n = np.linalg.norm(p)
        if n > 1e-8:
            p /= n
        desc[i] = p.reshape(-1)
    return desc, pts


def create_sift():
    return cv2.SIFT_create()


def sift_on_harris(gray_u8, points_rc, keypoint_size):
    # convert each harris point to a cv2 keypoint, then compute sift descriptors at those locations.
    sift = create_sift()
    kps = [cv2.KeyPoint(float(c), float(r), float(keypoint_size)) for r, c in points_rc]
    kps, desc = sift.compute(gray_u8, kps)
    out_points = np.array([[int(round(k.pt[1])), int(round(k.pt[0]))] for k in kps], dtype=int)
    return out_points, desc.astype(np.float32)


def harris_sift_points_desc(image, sigma, k, threshold_rel, min_distance, max_points, keypoint_size):
    # run the full extraction pipeline: grayscale conversion, harris detection, sift descriptors.
    gray = to_gray_float(image)
    gray_u8 = to_gray_u8(image)
    points = detect_harris_points(gray, sigma, k, threshold_rel, min_distance, max_points)
    return sift_on_harris(gray_u8, points, keypoint_size)


def pairs_to_xy(points1_rc, points2_rc, pairs):
    # index the matched points and flip from row/col to x/y order for geometric fitting.
    src = points1_rc[pairs[:, 0]][:, ::-1].astype(np.float64)
    dst = points2_rc[pairs[:, 1]][:, ::-1].astype(np.float64)
    return src, dst


def as_rgb_float(image):
    # ensure the image is 3-channel float rgb for consistent plotting.
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    return img_as_float(image)


def fit_affine(src, dst):
    # build a linear system and solve with least squares to estimate the affine matrix.
    n = len(src)
    if n < 3:
        return None

    x, y = src[:, 0], src[:, 1]
    u, v = dst[:, 0], dst[:, 1]

    A = np.zeros((2 * n, 6), dtype=np.float64)
    A[0::2, :3] = np.column_stack((x, y, np.ones(n)))
    A[1::2, 3:] = np.column_stack((x, y, np.ones(n)))
    b = np.empty(2 * n, dtype=np.float64)
    b[0::2], b[1::2] = u, v

    p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]]], dtype=np.float64)


def residuals_sq(M, src, dst):
    # apply the affine transform to src and return the squared distance to dst for each point.
    pred = np.column_stack((src, np.ones(len(src)))) @ M.T
    d = pred - dst
    return np.sum(d * d, axis=1)


def run_ransac(src, dst, sample_size, iterations, threshold, rng):
    # repeatedly sample random point pairs, fit an affine model, and keep the one with the most inliers.
    # after the loop, refit once on all inliers to get a more accurate final model.
    n, thr2 = len(src), threshold * threshold
    if n < sample_size:
        return None

    best = None
    for _ in range(iterations):
        idx = rng.choice(n, size=sample_size, replace=False)
        M = fit_affine(src[idx], dst[idx])
        r2 = residuals_sq(M, src, dst)
        mask = r2 <= thr2
        inliers = int(mask.sum())
        if inliers == 0:
            continue
        avg = float(r2[mask].mean())

        if best is None or inliers > best["inliers"] or (inliers == best["inliers"] and avg < best["avg"]):
            best = {"M": M, "mask": mask, "inliers": inliers, "avg": avg}

    if best is None:
        return None

    M_ref = fit_affine(src[best["mask"]], dst[best["mask"]])
    if M_ref is None:
        return best
    r2 = residuals_sq(M_ref, src, dst)
    mask = r2 <= thr2
    refined = {"M": M_ref, "mask": mask, "inliers": int(mask.sum()), "avg": float(r2[mask].mean())}
    if refined["inliers"] >= best["inliers"]:
        best = refined

    return best
