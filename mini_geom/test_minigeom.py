"""Self-contained tests for mini_geom (calibration / PnP / optical flow).

Everything is validated against synthetic ground truth, so no data files and no
third-party CV library are needed:

    python -m mini_geom.test_minigeom
"""

import numpy as np

from .calib import calibrate, checkerboard, rectified_points, synth_views
from .flow import (affine_lk, bilinear, gaussian_pyramid, min_safe_margin,
                   pyramid_lk)
from .geometry import project as geom_project, rodrigues
from .pnp import (dlt_pose, p3p, ransac_pnp, refine_pose, reprojection_error,
                  rotation_angle, solve_pnp)

K_TRUE = np.array([[820.0, 0.0, 322.0],
                   [0.0, 815.0, 238.0],
                   [0.0, 0.0, 1.0]])
DIST_TRUE = (-0.18, 0.04)


# ----------------------------------------------------------------- calibration

def test_calibration_noiseless():
    obj, views, _ = synth_views(K_TRUE, DIST_TRUE, n_views=6, noise=0.0, seed=1)
    res = calibrate([obj] * len(views), views)
    K = res["K"]
    assert abs(K[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0] < 1e-3, K[0, 0]
    assert abs(K[1, 1] - K_TRUE[1, 1]) / K_TRUE[1, 1] < 1e-3, K[1, 1]
    assert abs(K[0, 2] - K_TRUE[0, 2]) < 0.5, K[0, 2]
    assert abs(K[1, 2] - K_TRUE[1, 2]) < 0.5, K[1, 2]
    assert res["rms"] < 0.05, res["rms"]
    assert abs(res["dist"][0] - DIST_TRUE[0]) < 0.01, res["dist"]
    print("  calibration (noiseless)      fx=%.3f fy=%.3f cx=%.3f cy=%.3f "
          "k1=%.4f rms=%.4f px"
          % (K[0, 0], K[1, 1], K[0, 2], K[1, 2], res["dist"][0], res["rms"]))


def test_calibration_noisy():
    obj, views, _ = synth_views(K_TRUE, DIST_TRUE, n_views=10, noise=0.3, seed=7)
    res = calibrate([obj] * len(views), views)
    K, dist = res["K"], res["dist"]
    fx_err = abs(K[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0]
    cx_err = abs(K[0, 2] - K_TRUE[0, 2])
    assert fx_err < 0.03, fx_err
    assert cx_err < 4.0, cx_err
    assert res["rms"] < 0.5, res["rms"]
    assert abs(dist[0] - DIST_TRUE[0]) < 0.08, dist
    print("  calibration (0.3 px noise)   fx err=%.3f%%  cx err=%.2f px  "
          "k1=%.4f (gt %.2f)  rms=%.4f px"
          % (fx_err * 100, cx_err, dist[0], DIST_TRUE[0], res["rms"]))


def test_undistortion_improves_reprojection():
    obj, views, poses = synth_views(K_TRUE, DIST_TRUE, n_views=4, noise=0.0,
                                    seed=3)
    res = calibrate([obj] * len(views), views)
    K_hat, dist_hat = res["K"], res["dist"]

    def rms(points):
        return float(np.mean([
            np.sqrt(np.mean((geom_project(K_hat, r, t, obj) - p) ** 2))
            for p, (r, t) in zip(points, poses)]))

    undistorted = rectified_points(views, K_hat, dist_hat)
    resid = rms(undistorted)
    naive = rms(views)
    assert resid < 0.5, resid
    assert naive > 10.0 * resid, (naive, resid)
    print("  undistort -> pinhole residual %.4f px vs %.2f px raw "
          "(%.0fx better)" % (resid, naive, naive / resid))


# ------------------------------------------------------------------------ PnP

def _pnp_scene(n=20, noise=0.0, seed=11):
    rng = np.random.default_rng(seed)
    obj = np.column_stack([rng.uniform(-1.0, 1.0, n),
                           rng.uniform(-0.8, 0.8, n),
                           rng.uniform(3.0, 6.0, n)])
    rvec = np.array([0.25, -0.4, 0.15])
    tvec = np.array([0.3, -0.2, 7.0])
    img = geom_project(K_TRUE, rvec, tvec, obj, None)
    if noise:
        img = img + rng.normal(0.0, noise, img.shape)
    return obj, img, rvec, tvec


def test_dlt_pose_noiseless():
    obj, img, rvec, tvec = _pnp_scene(noise=0.0)
    r_hat, t_hat = dlt_pose(obj, img, K_TRUE)
    ang = rotation_angle(rodrigues(rvec), rodrigues(r_hat))
    terr = np.linalg.norm(t_hat - tvec)
    assert ang < 1e-4, ang
    assert terr < 1e-4, terr
    print("  DLT PnP (noiseless)          rot err=%.2e deg  trans err=%.2e"
          % (ang, terr))


def test_dlt_pose_noisy_and_refined():
    obj, img, rvec, tvec = _pnp_scene(n=30, noise=0.5)
    r0, t0 = dlt_pose(obj, img, K_TRUE)
    rms0 = reprojection_error(obj, img, K_TRUE, r0, t0)
    r1, t1, _, rms1 = refine_pose(obj, img, K_TRUE, r0, t0)
    ang = rotation_angle(rodrigues(rvec), rodrigues(r1))
    assert rms1 < rms0, (rms0, rms1)
    assert ang < 0.5, ang
    assert np.linalg.norm(t1 - tvec) / np.linalg.norm(tvec) < 0.02
    print("  DLT+LM (0.5 px noise)        rms %.4f -> %.4f px  rot err=%.4f deg"
          % (rms0, rms1, ang))


def test_p3p_minimal():
    obj, img, rvec, tvec = _pnp_scene(n=3, noise=0.0, seed=5)
    cands = p3p(obj, img, K_TRUE)
    assert cands, "P3P returned no solution"
    assert 1 <= len(cands) <= 4, len(cands)
    angles = [rotation_angle(rodrigues(rvec), rodrigues(r)) for r, _, _ in cands]
    best = min(angles)
    assert best < 1e-6, best
    print("  P3P (3 points, noiseless)    %d candidate(s), best rot err=%.2e deg"
          % (len(cands), best))


def test_p3p_with_refinement():
    obj, img, rvec, tvec = _pnp_scene(n=40, noise=0.3, seed=9)
    res = solve_pnp(obj, img, K_TRUE, use_p3p=True)
    ang = rotation_angle(rodrigues(rvec), rodrigues(res["rvec"]))
    assert ang < 0.5, ang
    assert res["rms"] < 0.5, res["rms"]
    print("  P3P + LM (40 pts, 0.3 px)    %d candidate(s) -> rot err=%.4f deg  "
          "rms=%.4f px" % (res["n_candidates"], ang, res["rms"]))


def test_ransac_pnp_with_outliers():
    obj, img, rvec, tvec = _pnp_scene(n=80, noise=0.4, seed=13)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(obj), 24, replace=False)        # 30% gross outliers
    img = img.copy()
    img[idx] += rng.uniform(-45.0, 45.0, (len(idx), 2))

    naive = solve_pnp(obj, img, K_TRUE, use_p3p=False)
    naive_ang = rotation_angle(rodrigues(rvec), rodrigues(naive["rvec"]))
    res = ransac_pnp(obj, img, K_TRUE, threshold=3.0, iters=400, seed=1)
    ang = rotation_angle(rodrigues(rvec), rodrigues(res["rvec"]))
    assert ang < 1.0, ang
    assert res["n_inliers"] >= 50, res["n_inliers"]
    assert ang < naive_ang, (ang, naive_ang)
    print("  RANSAC PnP (30%% outliers)    %d/%d inliers  rot err %.4f deg "
          "(plain least squares: %.2f deg)"
          % (res["n_inliers"], len(obj), ang, naive_ang))


# ---------------------------------------------------------------- optical flow

def _texture(h=200, w=260, seed=0, blur=1.4):
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, 1.0, (h, w))
    from .flow import gaussian_kernel, _convolve1d
    k = gaussian_kernel(blur)
    img = _convolve1d(_convolve1d(img, k, 0), k, 1)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return img


def _shift(img, dx, dy):
    h, w = img.shape
    ys, xs = np.mgrid[0:h, 0:w]
    val, _ = bilinear(img, xs - dx, ys - dy)
    return val


def _affine_warp(img, A, t):
    h, w = img.shape
    ys, xs = np.mgrid[0:h, 0:w]
    Ainv = np.linalg.inv(A)
    sx = Ainv[0, 0] * (xs - t[0]) + Ainv[0, 1] * (ys - t[1])
    sy = Ainv[1, 0] * (xs - t[0]) + Ainv[1, 1] * (ys - t[1])
    val, _ = bilinear(img, sx, sy)
    return val


def _kp_grid(h, w, step=40, margin=30):
    return np.array([(y, x) for y in range(margin, h - margin, step)
                     for x in range(margin, w - margin, step)], dtype=float)


def test_pyramid_shapes():
    img = _texture(160, 200)
    pyr = gaussian_pyramid(img, 4)
    assert [p.shape for p in pyr] == [(160, 200), (80, 100), (40, 50), (20, 25)]
    print("  gaussian pyramid             %s" % ([p.shape[0] for p in pyr],))


def test_lk_small_translation():
    img1 = _texture()
    gt = np.array([-1.7, 2.3])                   # (dy, dx), sub-pixel
    img2 = _shift(img1, gt[1], gt[0])
    pts = _kp_grid(*img1.shape)
    pts2, d = pyramid_lk(img1, img2, pts, levels=1, win=9)
    err = np.linalg.norm(d - gt, axis=1).mean()
    assert err < 0.05, err
    print("  LK single level, gt (2.3,-1.7)  mean err=%.5f px" % err)


def test_lk_large_translation():
    img1 = _texture(300, 380)
    gt = np.array([-8.4, 12.6])                  # needs a pyramid
    img2 = _shift(img1, gt[1], gt[0])
    levels, win = 4, 15
    pts = _kp_grid(*img1.shape, step=32, margin=min_safe_margin(levels, win))
    _, d1 = pyramid_lk(img1, img2, pts, levels=1, win=win)
    _, d2 = pyramid_lk(img1, img2, pts, levels=levels, win=win)
    e1 = np.linalg.norm(d1 - gt, axis=1).mean()
    e2 = np.linalg.norm(d2 - gt, axis=1).mean()
    assert e2 < 0.1, e2
    assert e2 < 0.05 * e1, (e1, e2)
    print("  LK pyramid, gt (12.6,-8.4)   single-level err=%.3f px  "
          "4-level err=%.4f px" % (e1, e2))


def _blobs(h=220, w=280, seed=3, n=60, rad=(6, 14)):
    """Structured texture: a sum of Gaussian blobs.

    Affine LK needs real 2-D structure; band-limited white noise has an
    autocorrelation so narrow that the six-parameter fit drifts to a
    degenerate (shrinking) solution.
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w))
    for _ in range(n):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        r = rng.uniform(*rad)
        img += rng.uniform(0.5, 1.0) * np.exp(
            -((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * r * r))
    return (img - img.min()) / (img.max() - img.min()) * 255.0


def test_lk_affine():
    img1 = _blobs()
    th = np.deg2rad(5.0)
    A_gt = np.array([[1.06 * np.cos(th), -1.06 * np.sin(th)],
                     [1.06 * np.sin(th), 1.06 * np.cos(th)]])
    t_gt = np.array([2.0, -1.5])
    img2 = _affine_warp(img1, A_gt, t_gt)
    levels, win = 3, 25
    pts = _kp_grid(*img1.shape, step=45, margin=min_safe_margin(levels, win))
    res = affine_lk(img1, img2, pts, win=win, levels=levels, iters=200)
    errs = np.array([np.linalg.norm(A - A_gt) / np.linalg.norm(A_gt)
                     for A, _ in res])
    med, good = float(np.median(errs)), float((errs < 0.05).mean())
    assert med < 0.02, med
    assert good >= 0.9, good
    print("  affine LK (5 deg, scale 1.06) median |A-A_gt|/|A_gt| = %.4f  "
          "%.0f%% of %d points within 5%%" % (med, good * 100, len(errs)))


# ----------------------------------------------------------------------- main

def main():
    print("mini_geom tests")
    print("[calibration]")
    test_calibration_noiseless()
    test_calibration_noisy()
    test_undistortion_improves_reprojection()
    print("[pnp]")
    test_dlt_pose_noiseless()
    test_dlt_pose_noisy_and_refined()
    test_p3p_minimal()
    test_p3p_with_refinement()
    test_ransac_pnp_with_outliers()
    print("[optical flow]")
    test_pyramid_shapes()
    test_lk_small_translation()
    test_lk_large_translation()
    test_lk_affine()
    print("all mini_geom tests passed")


if __name__ == "__main__":
    main()
