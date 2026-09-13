"""Zhang's planar camera calibration, pure numpy.

Pipeline
    1. per-view homography           -> normalised DLT
    2. closed-form intrinsics        -> orthogonality constraints on B = K^-T K^-1
    3. per-view extrinsics           -> decompose K^-1 H, orthonormalise R
    4. joint refinement              -> Levenberg-Marquardt over
                                        (fx, fy, cx, cy, k1, k2) + every view pose
"""

import numpy as np

from .geometry import (levenberg_marquardt, numeric_jacobian, orthonormalize,
                       project, rodrigues, rodrigues_inv, undistort_points)


# ------------------------------------------------------------- homography DLT

def _normalize(pts):
    """Isotropic normalisation: centroid -> origin, mean distance -> sqrt(2)."""
    pts = np.asarray(pts, dtype=np.float64)
    centroid = pts.mean(0)
    d = np.linalg.norm(pts - centroid, axis=1).mean()
    s = np.sqrt(2.0) / (d + 1e-12)
    T = np.array([[s, 0.0, -s * centroid[0]],
                  [0.0, s, -s * centroid[1]],
                  [0.0, 0.0, 1.0]])
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ hom.T).T[:, :2], T


def homography_dlt(src, dst):
    """Normalised DLT homography mapping src (N,2) -> dst (N,2)."""
    s, Ts = _normalize(src)
    d, Td = _normalize(dst)
    n = len(s)
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = s[i]
        u, v = d[i]
        A[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H @ Ts
    return H / H[2, 2]


def _v_ij(H, i, j):
    """Row vector v_ij such that v_ij @ b = (B)_ij for b = vector(B)."""
    h = H
    return np.array([
        h[0, i - 1] * h[0, j - 1],
        h[0, i - 1] * h[1, j - 1] + h[1, i - 1] * h[0, j - 1],
        h[1, i - 1] * h[1, j - 1],
        h[2, i - 1] * h[0, j - 1] + h[0, i - 1] * h[2, j - 1],
        h[2, i - 1] * h[1, j - 1] + h[1, i - 1] * h[2, j - 1],
        h[2, i - 1] * h[2, j - 1],
    ])


def intrinsics_from_homographies(Hs):
    """Closed-form K from >= 3 planar homographies (Zhang 2000, eq. 5-8)."""
    V = []
    for H in Hs:
        V.append(_v_ij(H, 1, 2))
        V.append(_v_ij(H, 1, 1) - _v_ij(H, 2, 2))
    _, _, Vt = np.linalg.svd(np.array(V))
    b = Vt[-1]
    B11, B12, B22, B13, B23, B33 = b
    denom = B11 * B22 - B12 * B12
    v0 = (B12 * B13 - B11 * B23) / denom
    lam = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11
    fx = np.sqrt(lam / B11)
    fy = np.sqrt(lam * B11 / denom)
    gamma = -B12 * fx * fx * fy / lam
    u0 = gamma * v0 / fy - B13 * fx * fx / lam
    return np.array([[fx, gamma, u0],
                     [0.0, fy, v0],
                     [0.0, 0.0, 1.0]])


def extrinsics_from_homography(H, K):
    """Recover (R, t) from one homography and the intrinsics."""
    Kinv = np.linalg.inv(K)
    h1, h2, h3 = Kinv @ H[:, 0], Kinv @ H[:, 1], Kinv @ H[:, 2]
    lam = 1.0 / (np.linalg.norm(h1) + 1e-15)
    if lam * h3[2] < 0:               # keep the board in front of the camera
        lam = -lam
    r1, r2, t = lam * h1, lam * h2, lam * h3
    R = orthonormalize(np.stack([r1, r2, np.cross(r1, r2)], axis=1))
    return rodrigues_inv(R), t


# --------------------------------------------------------------------- solvers

def _pack_params(K, dist, rvecs, tvecs, with_intrinsics=True):
    p = []
    if with_intrinsics:
        p += [K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2], dist[0], dist[1]]
    for r, t in zip(rvecs, tvecs):
        p.extend(list(r) + list(t))
    return np.array(p, dtype=np.float64)


def _unpack_params(p, n_views, with_intrinsics=True):
    if with_intrinsics:
        K = np.array([[p[0], p[2], p[3]],
                      [0.0, p[1], p[4]],
                      [0.0, 0.0, 1.0]])
        dist = (p[5], p[6])
        rest = p[7:]
    else:
        K = dist = None
        rest = p
    rvecs = [rest[6 * i:6 * i + 3] for i in range(n_views)]
    tvecs = [rest[6 * i + 3:6 * i + 6] for i in range(n_views)]
    return K, dist, rvecs, tvecs


def _residual(obj_pts, img_pts, n_views, fixed_k, fixed_dist):
    """Factory for the joint reprojection residual used by LM.

    When fixed_k is given only the view poses are free, which keeps the
    Jacobian full rank instead of leaving dead columns for the frozen block.
    """
    free_intr = fixed_k is None

    def f(p):
        K, dist, rvecs, tvecs = _unpack_params(p, n_views, free_intr)
        if not free_intr:
            K, dist = fixed_k, fixed_dist
        res = []
        for X, x, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
            res.append((project(K, r, t, X, dist) - x).ravel())
        return np.concatenate(res)
    return f


def calibrate(obj_points, img_points, refine=True, free_distortion=True,
              max_iter=200):
    """Full calibration.

    obj_points / img_points: equal-length lists, each (N, 3) world points on a
    z = 0 plane and their (N, 2) pixel observations.

    Returns dict(K, dist, rvecs, tvecs, rms, n_views).
    """
    obj_pts = [np.asarray(o, dtype=np.float64) for o in obj_points]
    img_pts = [np.asarray(p, dtype=np.float64) for p in img_points]
    n_views = len(obj_pts)
    if n_views < 3:
        raise ValueError("planar calibration needs at least 3 views")

    Hs = [homography_dlt(o[:, :2], p) for o, p in zip(obj_pts, img_pts)]
    K = intrinsics_from_homographies(Hs)
    dist = (0.0, 0.0)
    poses = [extrinsics_from_homography(H, K) for H in Hs]
    rvecs = [r for r, _ in poses]
    tvecs = [t for _, t in poses]

    if refine:
        p0 = _pack_params(K, dist, rvecs, tvecs, free_distortion)
        if free_distortion:
            res = _residual(obj_pts, img_pts, n_views, None, None)
        else:
            res = _residual(obj_pts, img_pts, n_views, K, dist)
        p = levenberg_marquardt(res, p0, jacobian=numeric_jacobian(res),
                                max_iter=max_iter)
        K, dist, rvecs, tvecs = _unpack_params(p, n_views, free_distortion)
        if not free_distortion:
            K = intrinsics_from_homographies(Hs)
            dist = (0.0, 0.0)

    e = _residual(obj_pts, img_pts, n_views, None, None)(
        _pack_params(K, dist, rvecs, tvecs))
    rms = float(np.sqrt(np.mean(e ** 2)))
    return {"K": K, "dist": dist, "rvecs": rvecs, "tvecs": tvecs,
            "rms": rms, "n_views": n_views}


def rectified_points(img_points, K, dist):
    """Undistorted pixel observations, ready for a linear (pinhole) solve."""
    return [undistort_points(p, K, dist) for p in img_points]


# ------------------------------------------------------------------- fixtures

def checkerboard(nx=9, ny=6, size=1.0):
    """(nx*ny, 3) object points of an inner-corner grid on the z = 0 plane."""
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    return np.stack([gx.ravel() * size, gy.ravel() * size,
                     np.zeros(nx * ny)], axis=1)


def synth_views(K, dist, n_views=6, nx=9, ny=6, noise=0.0, seed=0,
                radius=4.0, depth=9.0):
    """Render a checkerboard from random poses; returns (obj, img, poses)."""
    rng = np.random.default_rng(seed)
    obj = checkerboard(nx, ny)
    views, poses = [], []
    for _ in range(n_views):
        rvec = np.deg2rad(rng.uniform(-25.0, 25.0, size=3))
        t = np.array([rng.uniform(-radius, radius),
                      rng.uniform(-radius, radius),
                      depth + rng.uniform(-1.5, 1.5)])
        img = project(K, rvec, t, obj, dist)
        if noise:
            img = img + rng.normal(0.0, noise, img.shape)
        views.append(img)
        poses.append((rvec, t))
    return obj, views, poses
