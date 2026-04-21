import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage import io

# steps
# 1) load image 1, image 2, and best affine matrix from step 5
# 2) convert affine 2x3 to homogeneous 3x3
# 3) invert it to map image 2 coordinates into image 1 coordinates
# 4) build a larger common canvas from transformed corners
# 5) warp both images into that common canvas
# 6) blend with feather weights to reduce visible seams


def to_rgb(image):
    # warping/blending is easier when images have same 3-channel shape.
    
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def transform_corners(H, w, h):
    # build 4 corner points and apply perspective transform.
    corners = np.array([[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]], dtype=np.float32)
    return cv2.perspectiveTransform(corners, H.astype(np.float64))[0]


def main():
    #  load affine -> invert map -> build canvas -> warp both -> feather blend.
    p = argparse.ArgumentParser(description="step 6: warp image 2 onto image 1")
    p.add_argument("--img1", default="mnt1.JPG")
    p.add_argument("--img2", default="mnt2.JPG")
    p.add_argument("--affine_npy", default="outputs/step5_ransac_affine.npy")
    p.add_argument("--out_warp", default="outputs/step6_warped_img2_canvas.png")
    p.add_argument("--out_overlay", default="outputs/step6_feather_blend.png")
    a = p.parse_args()

    img1 = to_rgb(io.imread(a.img1))
    img2 = to_rgb(io.imread(a.img2))
    A = np.load(a.affine_npy)

    # extend the 2x3 affine to a 3x3 matrix and invert it to warp image2 onto image1's frame.
    H_1_to_2 = np.vstack([A, [0.0, 0.0, 1.0]]).astype(np.float64)
    H_2_to_1 = np.linalg.inv(H_1_to_2)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # compute panorama bounds using corners from image 1 and transformed image 2
    c1 = transform_corners(np.eye(3), w1, h1)
    c2 = transform_corners(H_2_to_1, w2, h2)
    all_c = np.vstack([c1, c2])

    min_x, min_y = np.floor(np.min(all_c, axis=0)).astype(int)
    max_x, max_y = np.ceil(np.max(all_c, axis=0)).astype(int)

    # translation to shift the canvas origin so all pixel coordinates are non-negative.
    tx, ty = -min_x, -min_y
    T = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
    out_w = int(max_x - min_x + 1)
    out_h = int(max_y - min_y + 1)

    warp1 = cv2.warpPerspective(img1, T, (out_w, out_h), flags=cv2.INTER_LINEAR)
    warped2 = cv2.warpPerspective(
        img2,
        T @ H_2_to_1,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # make binary masks, compute distance weights, blend weighted pixels.
    m1 = cv2.warpPerspective(np.full((h1, w1), 255, dtype=np.uint8), T, (out_w, out_h), flags=cv2.INTER_NEAREST) > 0
    m2 = cv2.warpPerspective(np.full((h2, w2), 255, dtype=np.uint8), T @ H_2_to_1, (out_w, out_h), flags=cv2.INTER_NEAREST) > 0
    d1 = cv2.distanceTransform((m1.astype(np.uint8) * 255), cv2.DIST_L2, 3).astype(np.float32)
    d2 = cv2.distanceTransform((m2.astype(np.uint8) * 255), cv2.DIST_L2, 3).astype(np.float32)

    denom = d1 + d2 + 1e-6
    w1f, w2f = d1 / denom, d2 / denom
    w1f[m1 & ~m2], w2f[m1 & ~m2] = 1.0, 0.0
    w1f[~m1 & m2], w2f[~m1 & m2] = 0.0, 1.0
    w1f[~m1 & ~m2], w2f[~m1 & ~m2] = 0.0, 0.0

    overlay = warp1.astype(np.float32) * w1f[..., None] + warped2.astype(np.float32) * w2f[..., None]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    out_warp = Path(a.out_warp)
    out_overlay = Path(a.out_overlay)
    out_warp.parent.mkdir(parents=True, exist_ok=True)
    out_overlay.parent.mkdir(parents=True, exist_ok=True)

    io.imsave(out_warp, warped2)
    io.imsave(out_overlay, overlay)

    print(f"loaded affine from: {a.affine_npy}")
    print("used inverse affine to map image2 -> image1 on a larger canvas")
    print(f"canvas size: {out_w} x {out_h}")
    print(f"saved warped image: {out_warp}")
    print(f"saved feather blend: {out_overlay}")


if __name__ == "__main__":
    main()
