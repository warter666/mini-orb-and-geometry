"""Perspective-n-Point solvers in pure numpy.

Two complementary solvers plus a non-linear refiner:
    dlt_pose   -- linear, needs >= 6 correspondences, no initial guess
    p3p        -- minimal, 3 correspondences, closed form via a Sylvester
                  resultant (up to 4 candidate poses, as P3P must give)
    refine_pose -- Levenberg-Marquardt on the reprojection error
"""

import numpy as np
from numpy.polynomial import Polynomial

from .geometry import (levenberg_marquardt, numeric_jacobian, orthonormalize,
                       project, rigid_transform, rodrigues, rodrigues_inv,
                       undistort_points)


def normalize_image_points(img_points, K):
    """Pixel observations -> unit bearing vectors (N, 3)."""
    Kinv = np.linalg.inv(K)
    hom = np.hstack([np.asarray(img_points, dtype=np.float64),
                     np.ones((len(img_points), 1))])
    rays = (Kinv @ hom.T).T
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def dlt_pose(obj_points, img_points, K, dist=None):
    """Linear PnP: solve the 3x4 projection matrix, then strip K.

    Requires at least 6 correspondences. Returns (rvec, tvec).
    """
    obj = np.asarray(obj_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(img_points, dtype=np.float64).reshape(-1, 2)
    if dist is not None:
        img = undistort_points(img, K, dist)
    if len(obj) < 6:
        raise ValueError("DLT needs at least 6 correspondences")

    rays = normalize_image_points(img, K)          # unused unit rays, kept
    hom = (np.linalg.inv(K) @
           np.hstack([img, np.ones((len(img), 1))]).T).T
    xn = hom[:, 0] / hom[:, 2]
    yn = hom[:, 1] / hom[:, 2]

    n = len(obj)
    A = np.zeros((2 * n, 12))
    for i in range(n):
        X, Y, Z = obj[i]
        x, y = xn[i], yn[i]
        A[2 * i] = [X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x]
        A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y]
    _, _, Vt = np.linalg.svd(A)
    M = Vt[-1].reshape(3, 4)

    # M ~ scale * [R | t]; the singular values of the 3x3 block give |scale|
    # and sign(det(U V^T)) tells us the sign of the scale.
    A3, b = M[:, :3], M[:, 3]
    U, S, Vt2 = np.linalg.svd(A3)
    lam = float(S.mean())
    R = U @ Vt2
    if np.linalg.det(R) < 0:
        lam = -lam
        R = -R                                     # det = +1 again
    t = b / lam
    _ = rays
    return rodrigues_inv(R), t


# --------------------------------------------------------------------- P3P

def _det_poly(M):
    """Determinant of a 4x4 matrix whose entries are Polynomials."""
    if len(M) == 1:
        return M[0][0]
    total = Polynomial([0.0])
    for j in range(len(M)):
        minor = [[M[i][k] for k in range(len(M)) if k != j]
                 for i in range(1, len(M))]
        term = M[0][j] * _det_poly(minor)
        total = total + term if j % 2 == 0 else total - term
    return total


def p3p(obj_points, img_points, K, dist=None):
    """Closed-form P3P (Gruenert-style resultant).

    P3P is inherently multi-valued: three correspondences can admit up to four
    distinct poses, all of which reproject exactly onto the three image points.
    Returns every valid candidate as (rvec, tvec, residual), best-conditioned
    first; the caller must disambiguate with a fourth point or a prior.
    """
    obj = np.asarray(obj_points, dtype=np.float64).reshape(-1, 3)[:3]
    img = np.asarray(img_points, dtype=np.float64).reshape(-1, 2)[:3]
    if dist is not None:
        img = undistort_points(img, K, dist)
    j = normalize_image_points(img, K)             # unit rays j1, j2, j3

    P1, P2, P3 = obj
    a = np.linalg.norm(P2 - P3)
    b = np.linalg.norm(P1 - P3)
    c = np.linalg.norm(P1 - P2)
    ca = float(j[1] @ j[2])
    cb = float(j[0] @ j[2])
    cg = float(j[0] @ j[1])

    # (A)  c^2 (1 + v^2 - 2 v cb) - b^2 (1 + u^2 - 2 u cg) = 0
    # (B)  a^2 (1 + u^2 - 2 u cg) - c^2 (u^2 + v^2 - 2 u v ca) = 0
    one_u = Polynomial([1.0, -2.0 * cg, 1.0])      # 1 + u^2 - 2 u cg
    A2 = Polynomial([c * c])
    A1 = Polynomial([-2.0 * c * c * cb])
    A0 = Polynomial([c * c]) - Polynomial([b * b]) * one_u
    B2 = Polynomial([-c * c])
    B1 = Polynomial([0.0, 2.0 * c * c * ca])
    B0 = Polynomial([a * a]) * one_u - Polynomial([0.0, 0.0, c * c])

    Syl = [[A2, A1, A0, Polynomial([0.0])],
           [Polynomial([0.0]), A2, A1, A0],
           [B2, B1, B0, Polynomial([0.0])],
           [Polynomial([0.0]), B2, B1, B0]]
    det = _det_poly(Syl).trim()
    coeffs = det.coef
    if len(coeffs) < 2:
        return []
    # numpy.polynomial stores ascending powers; np.roots wants descending.
    scale = np.max(np.abs(coeffs))
    while len(coeffs) > 1 and abs(coeffs[-1]) <= 1e-14 * scale:
        coeffs = coeffs[:-1]

    candidates = []
    for root in np.roots(coeffs[::-1]):
        if abs(root.imag) > 1e-8:
            continue
        u = float(root.real)
        if u <= 0.0:
            continue
        denom = 1.0 + u * u - 2.0 * u * cg
        if denom <= 1e-12:
            continue
        s1 = np.sqrt(c * c / denom)
        s2 = u * s1
        # s3 from the second equation: s3^2 - 2 s1 cb s3 + (s1^2 - b^2) = 0
        disc = s1 * s1 * (cb * cb - 1.0) + b * b
        if disc < -1e-9:
            continue
        disc = max(disc, 0.0)
        for s3 in (s1 * cb + np.sqrt(disc), s1 * cb - np.sqrt(disc)):
            if s3 <= 1e-9:
                continue
            depth = np.array([s1, s2, s3])
            if np.any(depth <= 0):
                continue
            cam = j * depth[:, None]
            R, t = rigid_transform(obj, cam)
            # Rank by how well the root satisfies the two governing equations,
            # measured relatively (the residual magnitudes are O(1) here).
            v = s3 / s1
            one_u_ = 1.0 + u * u - 2.0 * u * cg
            one_v_ = 1.0 + v * v - 2.0 * v * cb
            term_a = c * c * one_v_
            term_b = b * b * one_u_
            term_c = a * a * one_u_
            term_d = c * c * (u * u + v * v - 2.0 * u * v * ca)
            res_a = abs(term_a - term_b) / max(abs(term_a), abs(term_b), 1e-12)
            res_b = abs(term_c - term_d) / max(abs(term_c), abs(term_d), 1e-12)
            candidates.append((rodrigues_inv(R), t, res_a + res_b))
    candidates.sort(key=lambda item: item[2])
    return candidates


# ------------------------------------------------------------------ refiner

def reprojection_error(obj_points, img_points, K, rvec, tvec, dist=None):
    """RMS pixel error of a pose."""
    obj = np.asarray(obj_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(img_points, dtype=np.float64).reshape(-1, 2)
    e = project(K, rvec, tvec, obj, dist) - img
    return float(np.sqrt(np.mean(e ** 2)))


def refine_pose(obj_points, img_points, K, rvec, tvec, dist=None,
                optimize_distortion=False, max_iter=100):
    """LM refinement of a pose, optionally of the radial distortion too.

    Returns (rvec, tvec, dist, rms).
    """
    obj = np.asarray(obj_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(img_points, dtype=np.float64).reshape(-1, 2)
    k1, k2 = dist if dist is not None else (0.0, 0.0)
    n_extra = 2 if optimize_distortion else 0

    def unpack(p):
        d = (p[6], p[7]) if n_extra else (k1, k2)
        return p[:3], p[3:6], d

    def residual(p):
        r, t, d = unpack(p)
        return (project(K, r, t, obj, d) - img).ravel()

    p0 = np.concatenate([np.asarray(rvec, dtype=np.float64).reshape(3),
                         np.asarray(tvec, dtype=np.float64).reshape(3),
                         [k1, k2][:n_extra]])
    p = levenberg_marquardt(residual, p0, jacobian=numeric_jacobian(residual),
                            max_iter=max_iter)
    rvec2, tvec2, dist2 = unpack(p)
    return rvec2, tvec2, dist2, reprojection_error(obj, img, K, rvec2, tvec2,
                                                   dist2)


def solve_pnp(obj_points, img_points, K, dist=None, use_p3p=None):
    """Convenience front door: pick a solver, then refine.

    use_p3p is auto-selected (True when fewer than 6 points are available).
    Every P3P candidate is refined against *all* correspondences and the one
    with the lowest reprojection error wins -- that is how the multi-valued
    P3P is disambiguated when a fourth point exists.
    Returns dict(rvec, tvec, dist, rms).
    """
    obj = np.asarray(obj_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(img_points, dtype=np.float64).reshape(-1, 2)
    if use_p3p is None:
        use_p3p = len(obj) < 6
    if use_p3p:
        cands = p3p(obj[:3], img[:3], K, dist)
        if not cands:
            raise RuntimeError("P3P found no valid configuration")
        best = None
        for r0, t0, _ in cands:
            r1, t1, d1, rms = refine_pose(obj, img, K, r0, t0, dist)
            if best is None or rms < best[3]:
                best = (r1, t1, d1, rms)
        return {"rvec": best[0], "tvec": best[1], "dist": best[2],
                "rms": best[3], "n_candidates": len(cands)}
    rvec, tvec = dlt_pose(obj, img, K, dist)
    rvec, tvec, dist2, rms = refine_pose(obj, img, K, rvec, tvec, dist)
    return {"rvec": rvec, "tvec": tvec, "dist": dist2, "rms": rms,
            "n_candidates": 1}


def rotation_angle(R1, R2):
    """Geodesic angle in degrees between two rotations."""
    R = np.asarray(R1).T @ np.asarray(R2)
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0,
                                              -1.0, 1.0))))


def ransac_pnp(obj_points, img_points, K, dist=None, threshold=3.0,
               iters=300, seed=0, min_inliers=6):
    """RANSAC over P3P, then a LM refit on the inliers.

    A handful of wrong correspondences is enough to drag a plain DLT/P3P fit
    off by tens of degrees, because the final least-squares step is not
    robust. P3P is the natural minimal solver here: 3 points give a finite
    (and small) number of candidate poses, each of which can be scored
    against all the others.

    Returns dict(rvec, tvec, dist, rms, inliers, n_inliers, n_candidates).
    """
    obj = np.asarray(obj_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(img_points, dtype=np.float64).reshape(-1, 2)
    n = len(obj)
    if n < 3:
        raise ValueError("RANSAC PnP needs at least 3 correspondences")
    rng = np.random.default_rng(seed)

    best = None
    n_candidates = 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        cands = p3p(obj[idx], img[idx], K, dist)
        n_candidates += len(cands)
        for r0, t0, _ in cands:
            err = np.linalg.norm(project(K, r0, t0, obj, dist) - img, axis=1)
            mask = err < threshold
            count = int(mask.sum())
            if best is None or count > best[0]:
                best = (count, mask, r0, t0)
        if best is not None and best[0] > 0.9 * n:
            break

    if best is None or best[0] < min_inliers:
        raise RuntimeError("RANSAC found only %s inliers"
                           % (0 if best is None else best[0]))
    count, mask, r0, t0 = best
    rvec, tvec, dist_out, _ = refine_pose(obj[mask], img[mask], K, r0, t0, dist)
    # One re-scoring pass with the refined pose tightens the inlier set.
    err = np.linalg.norm(project(K, rvec, tvec, obj, dist_out) - img, axis=1)
    mask = err < threshold
    rvec, tvec, dist_out, rms = refine_pose(obj[mask], img[mask], K,
                                            rvec, tvec, dist_out)
    return {"rvec": rvec, "tvec": tvec, "dist": dist_out, "rms": rms,
            "inliers": mask, "n_inliers": int(mask.sum()),
            "n_candidates": n_candidates}
