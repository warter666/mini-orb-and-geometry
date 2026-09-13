"""End-to-end demo: mini_orb features + mini_geom PnP on a synthetic scene.

A textured plane lies on z = 0. Frame 1 is the world reference camera; frame 2
sits at an unknown pose. The pipeline is the classic SfM front half:

    render two views -> ORB detect/describe -> Hamming match
    -> back-project frame-1 keypoints onto the known plane
    -> PnP (DLT + LM) against frame-2 pixels -> compare with ground truth

Run:  python -m demo_pipeline
"""

import numpy as np

from mini_geom.calib import project
from mini_geom.flow import _convolve1d, bilinear, gaussian_kernel
from mini_geom.geometry import rodrigues
from mini_geom.pnp import ransac_pnp, reprojection_error, rotation_angle
from mini_orb.features import detect_and_describe, match

K = np.array([[700.0, 0.0, 320.0],
              [0.0, 700.0, 240.0],
              [0.0, 0.0, 1.0]])
IMAGE = (480, 640)                       # h, w
TEXTURE_SIZE = 640
PLANE_HALF = 4.0                         # world extent of the plane: [-4, 4]
POSE1 = (np.zeros(3), np.array([0.0, 0.0, 4.0]))     # world reference camera


def make_texture(n=TEXTURE_SIZE, seed=17, sigma=1.0):
    """Band-limited noise texture.

    Bandwidth matters: FAST-9 needs a pixel difference above the threshold on
    a radius-3 circle, so a very smooth texture (large Gaussian blobs) yields
    a near-zero corner count no matter how high the global contrast is.
    """
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, 1.0, (n, n))
    k = gaussian_kernel(sigma)
    img = _convolve1d(_convolve1d(img, k, 0), k, 1)
    img = (img - img.min()) / (img.max() - img.min())
    return np.clip(img, 0.0, 1.0) * 255.0


def render_plane(texture, rvec, tvec, size=IMAGE, plane_half=PLANE_HALF):
    """Ray-cast a textured z = 0 plane through every pixel (pinhole model).

    Note there is deliberately no vertical flip: world +y maps to growing
    image rows through the camera model itself, so texture row order and
    image row order stay consistent.
    """
    h, w = size
    ys, xs = np.mgrid[0:h, 0:w]
    hom = np.stack([xs, ys, np.ones_like(xs)], axis=-1).astype(np.float64)
    rays = hom @ np.linalg.inv(K).T                     # camera-frame rays
    R = rodrigues(rvec)
    dirs = rays @ R                                     # world-frame directions
    C = -R.T @ tvec                                     # camera centre
    lam = -C[2] / dirs[..., 2]                          # intersect z = 0
    X = C[None, None, :] + lam[..., None] * dirs
    n = texture.shape[0] - 1
    u = (X[..., 0] + plane_half) / (2 * plane_half) * n
    v = (X[..., 1] + plane_half) / (2 * plane_half) * n
    img, valid = bilinear(texture, u, v)
    inside = (valid & (lam > 0)
              & (np.abs(X[..., 0]) < plane_half)
              & (np.abs(X[..., 1]) < plane_half))
    return np.where(inside, img, 0.0)


def backproject_to_plane(pixels, rvec, tvec, plane_half=PLANE_HALF, K_=K):
    """Pixels of a known pose -> world points on the z = 0 plane."""
    hom = np.hstack([pixels, np.ones((len(pixels), 1))])
    rays = hom @ np.linalg.inv(K_).T
    R = rodrigues(rvec)
    dirs = rays @ R
    C = -R.T @ tvec
    lam = -C[2] / dirs[:, 2]
    return C[None, :] + lam[:, None] * dirs


def main():
    texture = make_texture()
    r1, t1 = POSE1
    r2 = np.deg2rad(np.array([3.0, -5.0, 2.0]))
    t2 = np.array([0.45, -0.30, 3.55])

    img1 = render_plane(texture, r1, t1)
    img2 = render_plane(texture, r2, t2)
    print("rendered frames %s / %s" % (img1.shape, img2.shape))

    d1, kp1, _ = detect_and_describe(img1, n=500, threshold=15)
    d2, kp2, _ = detect_and_describe(img2, n=500, threshold=15)
    print("keypoints: frame1 %d, frame2 %d" % (len(kp1), len(kp2)))

    matches = match(d1, d2, ratio=0.85, max_dist=80)
    print("mutual-nearest matches: %d" % len(matches))
    if len(matches) < 12:
        raise RuntimeError("not enough matches to run PnP")

    idx1 = np.array([i for i, _, _ in matches], dtype=int)
    idx2 = np.array([j for _, j, _ in matches], dtype=int)
    pts1 = kp1[idx1][:, ::-1].astype(np.float64)        # (x, y)
    pts2 = kp2[idx2][:, ::-1].astype(np.float64)

    world = backproject_to_plane(pts1, r1, t1)
    check = np.linalg.norm(project(K, r2, t2, world) - pts2, axis=1)
    print("correspondence quality vs ground truth: %.0f%% within 3 px "
          "(median %.2f px, RMS %.2f px -- the outliers are what RANSAC "
          "is for)" % (100.0 * (check < 3).mean(), np.median(check),
                       np.sqrt(np.mean(check ** 2))))

    # The target is planar, which makes the 3x4 DLT rank deficient, and the
    # match set carries outliers, so RANSAC over the minimal P3P solver is
    # the correct front end.
    res = ransac_pnp(world, pts2, K, threshold=3.0, iters=400, seed=1)
    print("RANSAC: %d/%d inliers kept" % (res["n_inliers"], len(world)))

    err_rot = rotation_angle(rodrigues(r2), rodrigues(res["rvec"]))
    err_t = np.linalg.norm(res["tvec"] - t2)
    gt_rms = reprojection_error(world[res["inliers"]], pts2[res["inliers"]],
                                K, r2, t2)
    print("PnP rms %.4f px on inliers (ground-truth pose gives %.4f px)"
          % (res["rms"], gt_rms))
    print("rotation error %.4f deg   translation error %.4f (|t| = %.2f)"
          % (err_rot, err_t, np.linalg.norm(t2)))

    assert err_rot < 1.5, err_rot
    assert err_t < 0.05 * np.linalg.norm(t2), err_t
    print("end-to-end ORB -> match -> RANSAC-PnP pipeline passed")


if __name__ == "__main__":
    main()
