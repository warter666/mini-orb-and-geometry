"""Lucas-Kanade tracking in pure numpy: Gaussian pyramids, translational and
affine (6-DOF) inverse-compositional-less forward-additive solvers.
"""

import numpy as np


# ------------------------------------------------------------------- resampling

def bilinear(img, xs, ys):
    """Bilinear sampling at float coordinates; out-of-range -> 0 (+ mask)."""
    h, w = img.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    wx = xs - x0
    wy = ys - y0
    val = (img[y0c, x0c] * (1 - wx) * (1 - wy)
           + img[y0c, x1c] * wx * (1 - wy)
           + img[y1c, x0c] * (1 - wx) * wy
           + img[y1c, x1c] * wx * wy)
    return np.where(valid, val, 0.0), valid


def _convolve1d(img, kernel, axis):
    r = len(kernel) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    padded = np.pad(img, pad, mode="edge")
    out = np.zeros_like(img, dtype=np.float64)
    for i, kv in enumerate(kernel):
        sl = [slice(None), slice(None)]
        sl[axis] = slice(i, i + img.shape[axis])
        out += kv * padded[tuple(sl)]
    return out


def gaussian_kernel(sigma=1.0):
    r = int(np.ceil(3.0 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-x * x / (2.0 * sigma * sigma))
    return k / k.sum()


def pyr_down(img, sigma=1.0):
    """Blur then decimate by two."""
    k = gaussian_kernel(sigma)
    b = _convolve1d(_convolve1d(img.astype(np.float64), k, 0), k, 1)
    return b[::2, ::2]


def gaussian_pyramid(img, levels, sigma=1.0):
    pyr = [img.astype(np.float64)]
    for _ in range(levels - 1):
        pyr.append(pyr_down(pyr[-1], sigma))
    return pyr


# ------------------------------------------------------------- translational LK

def _patch_offsets(win):
    r = win // 2
    dy, dx = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1),
                         indexing="ij")
    return dy.ravel().astype(np.float64), dx.ravel().astype(np.float64)


def lk_solve(img1, img2, pts, d0, win=7, iters=30, eps=0.01):
    """One pyramid level of translational LK. pts and d0 are in this level's
    coordinates; returns the refined displacement d (N, 2) as (dy, dx).

    The Hessian is rebuilt at the current warp position on every iteration
    (original forward-additive form). Freezing it at the template position,
    as many textbook snippets do, shrinks the convergence radius to well
    under a pixel and makes coarse pyramid levels useless.
    """
    d = np.array(d0, dtype=np.float64, copy=True)
    oy, ox = _patch_offsets(win)
    r = win // 2
    gy2, gx2 = np.gradient(img2)
    min_eig = 1e-8

    for k in range(len(pts)):
        y, x = pts[k]
        if not (r <= y < img1.shape[0] - r and r <= x < img1.shape[1] - r):
            continue
        iy = (y + oy).astype(np.int64)
        ix = (x + ox).astype(np.int64)
        I1 = img1[iy, ix]

        for _ in range(iters):
            sx = x + ox + d[k, 1]
            sy = y + oy + d[k, 0]
            I2, valid = bilinear(img2, sx, sy)
            if valid.mean() < 0.8:
                break
            Ix, _ = bilinear(gx2, sx, sy)
            Iy, _ = bilinear(gy2, sx, sy)
            e = I2 - I1
            cost = float(e @ e)
            # H and b are ordered (dy, dx) to match the displacement layout.
            H = np.array([[np.sum(Iy * Iy), np.sum(Ix * Iy)],
                          [np.sum(Ix * Iy), np.sum(Ix * Ix)]])
            if np.linalg.eigvalsh(H)[0] < min_eig:
                break
            b = np.array([np.sum(Iy * e), np.sum(Ix * e)])
            step = -np.linalg.solve(H + 1e-9 * np.eye(2), b)
            # Backtracking: a raw Newton step can overshoot on coarse levels,
            # so halve it until the SSD actually drops.
            for _ in range(8):
                cand = d[k] + step
                I2c, valc = bilinear(img2, x + ox + cand[1], y + oy + cand[0])
                if valc.mean() >= 0.8 and float((I2c - I1) @ (I2c - I1)) < cost:
                    break
                step = step * 0.5
            d[k] += step
            if np.linalg.norm(step) < eps:
                break
    return d


def min_safe_margin(levels, win):
    """Smallest border margin that keeps a point inside *every* pyramid level.

    A point at image margin m sits at m / 2^(levels-1) on the coarsest level,
    which must still clear the patch half-width. Points closer to the border
    are silently skipped on the coarse levels and never recover.
    """
    return (win // 2 + 1) * 2 ** (levels - 1)


def pyramid_lk(img1, img2, pts, levels=3, win=7, iters=30, eps=0.01):
    """Coarse-to-fine translational LK.

    Border note: a point needs margin >= min_safe_margin(levels, win) pixels
    from every image edge, otherwise it cannot be tracked on the coarse levels.
    Returns (pts2, displacements) in original image coordinates, as (dy, dx).
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    pyr1 = gaussian_pyramid(img1, levels)
    pyr2 = gaussian_pyramid(img2, levels)
    d = np.zeros_like(pts)
    for lvl in range(levels - 1, -1, -1):
        scale = float(2 ** lvl)
        d = lk_solve(pyr1[lvl], pyr2[lvl], pts / scale, d, win, iters, eps)
        if lvl > 0:
            d = d * 2.0
    return pts + d, d


# ------------------------------------------------------------------- affine LK

def _affine_level(img1, img2, pts, A_init, t_init, win, iters, eps, damping):
    """One pyramid level of affine LK. A_init/t_init are this level's guesses
    (t in this level's pixels); returns refined (list(A), list(t))."""
    oy, ox = _patch_offsets(win)
    r = win // 2
    gy2, gx2 = np.gradient(img2)
    As, ts = [], []

    for k, (y, x) in enumerate(pts):
        A0 = A_init[k]
        t0 = t_init[k]
        if not (r <= y < img1.shape[0] - r and r <= x < img1.shape[1] - r):
            As.append(A0)
            ts.append(t0)
            continue
        # Sample the template bilinearly too: on coarse pyramid levels the
        # point coordinates are fractional and integer truncation would bias
        # the estimate.
        I1, _ = bilinear(img1, x + ox, y + oy)

        def warp(p):
            sx = x + (1.0 + p[0]) * ox + p[1] * oy + p[4]
            sy = y + p[2] * ox + (1.0 + p[3]) * oy + p[5]
            return sx, sy

        p = np.array([A0[0, 0] - 1.0, A0[0, 1], A0[1, 0], A0[1, 1] - 1.0,
                      t0[0], t0[1]])
        I2_0 = bilinear(img2, *warp(p))[0]
        cost = float((I2_0 - I1) @ (I2_0 - I1))
        for _ in range(iters):
            sx, sy = warp(p)
            I2, valid = bilinear(img2, sx, sy)
            if valid.mean() < 0.8:
                break
            Ix, _ = bilinear(gx2, sx, sy)
            Iy, _ = bilinear(gy2, sx, sy)
            J = np.stack([Ix * ox, Ix * oy, Iy * ox, Iy * oy, Ix, Iy], axis=1)
            H = J.T @ J
            # Six parameters need real 2-D structure; on a nearly flat or
            # one-directional patch the system is ill-posed, so bail out.
            if np.linalg.cond(H) > 1e8:
                break
            e = I2 - I1
            g = J.T @ e
            step = -np.linalg.solve(H + damping * np.trace(H) * np.eye(6), g)
            for _ in range(8):
                I2c, valc = bilinear(img2, *warp(p + step))
                if valc.mean() >= 0.8 and float((I2c - I1) @ (I2c - I1)) < cost:
                    break
                step = step * 0.5
            p = p + step
            I2_acc = bilinear(img2, *warp(p))[0]
            cost = float((I2_acc - I1) @ (I2_acc - I1))
            if np.linalg.norm(step) < eps:
                break
        As.append(np.array([[1.0 + p[0], p[1]], [p[2], 1.0 + p[3]]]))
        ts.append(p[4:6].copy())
    return As, ts


def affine_lk(img1, img2, pts, win=15, iters=50, eps=1e-4, damping=1e-3,
              levels=3):
    """Per-keypoint 6-DOF affine warping, coarse-to-fine.

    Parameterisation: p' = A (p - c) + c + t with A = I + [[p0, p1], [p2, p3]]
    and t = (p4, p5). A single level only converges for roughly 1 degree of
    rotation, so the pose is bootstrapped on a Gaussian pyramid. As in
    lk_solve, the Jacobian is rebuilt at the current warp on every iteration
    and each step is backtracked on SSD.
    Returns a list of (A, t) with A a 2x2 matrix, in original coordinates.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    pyr1 = gaussian_pyramid(img1, levels)
    pyr2 = gaussian_pyramid(img2, levels)
    As = [np.eye(2) for _ in range(len(pts))]
    ts = [np.zeros(2) for _ in range(len(pts))]
    for lvl in range(levels - 1, -1, -1):
        scale = float(2 ** lvl)
        As, ts = _affine_level(pyr1[lvl], pyr2[lvl], pts / scale, As, ts,
                               win, iters, eps, damping)
        if lvl > 0:
            ts = [t * 2.0 for t in ts]
    return list(zip(As, ts))


def track_feature_errors(img1, img2, pts, pts_gt):
    """Diagnostic: end-point error of a translational track."""
    err = np.asarray(pts_gt, dtype=np.float64) - np.asarray(pts, dtype=np.float64)
    return float(np.mean(np.linalg.norm(err, axis=1)))
