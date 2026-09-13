"""Shared geometry primitives: Lie helpers, rigid alignment, projection and a
Levenberg-Marquardt optimiser. Pure numpy, no scipy.
"""

import numpy as np


# ------------------------------------------------------------------ Lie group

def rodrigues(rvec):
    """Rotation vector (axis * angle) -> 3x3 rotation matrix."""
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-14:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rodrigues_inv(R):
    """3x3 rotation matrix -> rotation vector."""
    R = np.asarray(R, dtype=np.float64)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(c))
    if theta < 1e-9:
        return np.zeros(3)
    if np.pi - theta < 1e-5:
        # Near pi the antisymmetric part vanishes; recover the axis from R + I.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        i = int(np.argmax(axis))
        axis = A[:, i] / (axis[i] + 1e-12)
        return axis / np.linalg.norm(axis) * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta / (2.0 * np.sin(theta)) * w


def orthonormalize(M):
    """Closest rotation matrix to M (Frobenius), determinant forced to +1."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def rigid_transform(src, dst):
    """Least-squares rigid transform taking src -> dst, both (N, 3)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


# --------------------------------------------------------------- camera model

def project(K, rvec, tvec, X, dist=None):
    """Perspective projection of (N, 3) world points with optional k1, k2."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    Xc = X @ rodrigues(rvec).T + np.asarray(tvec, dtype=np.float64).reshape(1, 3)
    z = Xc[:, 2]
    x = Xc[:, 0] / z
    y = Xc[:, 1] / z
    if dist is not None:
        r2 = x * x + y * y
        rad = 1.0 + dist[0] * r2 + dist[1] * r2 * r2
        x, y = x * rad, y * rad
    u = K[0, 0] * x + K[0, 1] * y + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.stack([u, v], axis=1)


def undistort_points(pts, K, dist, iters=12):
    """Iterative inverse of the radial model; identity when dist is None."""
    if dist is None or (dist[0] == 0.0 and dist[1] == 0.0):
        return np.asarray(pts, dtype=np.float64).copy()
    k1, k2 = dist
    pts = np.asarray(pts, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xd = (pts[:, 0] - cx) / fx
    yd = (pts[:, 1] - cy) / fy
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        rad = 1.0 + k1 * r2 + k2 * r2 * r2
        x, y = xd / rad, yd / rad
    return np.stack([x * fx + cx, y * fy + cy], axis=1)


# ---------------------------------------------------------------- optimisation

def numeric_jacobian(f, rel_step=1e-7):
    """Central-difference Jacobian factory for a residual f: R^n -> R^m."""
    def jac(x):
        cols = []
        for i in range(len(x)):
            h = rel_step * max(1.0, abs(x[i]))
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            cols.append((f(xp) - f(xm)) / (2.0 * h))
        return np.stack(cols, axis=1)
    return jac


def levenberg_marquardt(residual, x0, jacobian=None, max_iter=100, tol=1e-12,
                        lam0=1e-3, callback=None):
    """Damped Gauss-Newton on ||residual(x)||^2 with adaptive damping."""
    x = np.array(x0, dtype=np.float64).reshape(-1)
    r = np.asarray(residual(x), dtype=np.float64).reshape(-1)
    cost = float(r @ r)
    lam = lam0
    jac = jacobian or numeric_jacobian(residual)
    for _ in range(max_iter):
        J = jac(x)
        g = J.T @ r
        if np.linalg.norm(g, np.inf) < tol:
            break
        A = J.T @ J
        diag = np.maximum(np.diag(A), 1e-12)
        accepted = False
        for _ in range(25):
            try:
                dx = np.linalg.solve(A + lam * np.diag(diag), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            x_new = x + dx
            r_new = np.asarray(residual(x_new), dtype=np.float64).reshape(-1)
            cost_new = float(r_new @ r_new)
            if cost_new < cost:
                x, r, cost = x_new, r_new, cost_new
                lam = max(lam * 0.3, 1e-12)
                accepted = True
                if callback is not None:
                    callback(x, cost)
                break
            lam *= 10.0
        if not accepted:
            break
    return x
