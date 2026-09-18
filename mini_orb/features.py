"""mini-ORB: FAST-9 corners + Harris scoring + intensity-centroid orientation +
steered BRIEF descriptors + Hamming matching, in pure numpy.
"""

import math

import numpy as np

CIRCLE = np.array([  # the 16 offsets of the FAST Bresenham circle, radius 3
    (0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3),
    (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3),
])
MARGIN = 18  # orientation patch (r=15) and BRIEF pattern (±15, smoothed) fit inside


def _integral(img):
    """Integral image with a zero row/col at the top-left."""
    return np.pad(img, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _box_sum(ii, r):
    """Sum over (2r+1)x(2r+1) windows, same shape as ii (minus border padding)."""
    h, w = ii.shape
    out = ii[2 * r + 1:, 2 * r + 1:] - ii[:-2 * r - 1, 2 * r + 1:] \
        - ii[2 * r + 1:, :-2 * r - 1] + ii[:-2 * r - 1, :-2 * r - 1]
    return out


def fast9_mask(img, threshold):
    """Vectorized FAST-9: returns (H, W) bool corner mask."""
    H, W = img.shape
    center = img[MARGIN:H - MARGIN, MARGIN:W - MARGIN]
    ys = np.arange(MARGIN, H - MARGIN)[:, None]
    xs = np.arange(MARGIN, W - MARGIN)[None, :]
    circle = np.stack([img[ys + dy, xs + dx] for dy, dx in CIRCLE])  # (16, h, w)
    brighter = circle > center + threshold
    darker = circle < center - threshold
    mask = np.zeros((H, W), dtype=bool)
    for flags in (brighter, darker):
        ext = np.concatenate([flags, flags], axis=0)  # (32, h, w)
        windows = np.lib.stride_tricks.sliding_window_view(ext, 9, axis=0)
        mask[MARGIN:H - MARGIN, MARGIN:W - MARGIN] |= (windows.sum(-1) == 9).any(0)
    return mask


def harris_scores(img):
    """Shi-Tomasi-ish Harris response via gradient products and box sums."""
    gy, gx = np.gradient(img.astype(np.float64))
    ii = _integral(gx * gx)
    ii_xy = _integral(gx * gy)
    ii_yy = _integral(gy * gy)
    # box radius 2 (5x5 window); box_sum output is offset by the integral padding
    r = 2
    sxx = _box_sum(ii, r)
    sxy = _box_sum(ii_xy, r)
    syy = _box_sum(ii_yy, r)
    resp = np.zeros(img.shape, dtype=np.float64)
    off = r  # box_sum[i] covers image rows i..i+2r, center i+r
    resp[off:off + sxx.shape[0], off:off + sxx.shape[1]] = \
        sxx * syy - sxy ** 2 - 0.04 * (sxx + syy) ** 2
    return resp


def detect_keypoints(img, n=300, threshold=15, nms_radius=5, max_candidates=2000):
    """FAST candidates -> Harris-ranked, NMS-suppressed keypoint list [(y, x)].

    Low FAST thresholds can yield tens of thousands of candidates; since only the
    top-n survivors matter, cap the NMS input at the highest-scoring
    `max_candidates` before the quadratic suppression loop.
    """
    H, W = img.shape
    mask = fast9_mask(img, threshold)
    resp = harris_scores(img)
    ys, xs = np.nonzero(mask)
    scores = resp[ys, xs]
    order = np.argsort(-scores)[:max_candidates]
    picked = []
    r2 = nms_radius * nms_radius
    for idx in order:
        y, x = ys[idx], xs[idx]
        if all((y - py) ** 2 + (x - px) ** 2 > r2 for py, px in picked):
            picked.append((y, x))
            if len(picked) == n:
                break
    return picked


_ORIENT_OFFSETS = None


def _orientation_offsets():
    global _ORIENT_OFFSETS
    if _ORIENT_OFFSETS is None:
        rr = np.arange(-15, 16)
        dy, dx = np.meshgrid(rr, rr, indexing="ij")
        inside = dy ** 2 + dx ** 2 <= 15 ** 2
        _ORIENT_OFFSETS = (dy[inside].astype(np.float64),
                           dx[inside].astype(np.float64))
    return _ORIENT_OFFSETS


def keypoint_orientations(img, keypoints):
    """Intensity centroid angle atan2(m01, m10) per keypoint."""
    H, W = img.shape
    dys, dxs = _orientation_offsets()
    angles = []
    for y, x in keypoints:
        patch = img[y + dys.astype(int), x + dxs.astype(int)]
        m10 = np.sum(dxs * patch)
        m01 = np.sum(dys * patch)
        angles.append(math.atan2(m01, m10))
    return np.array(angles)


def _make_pattern(n_bits=256, seed=1234):
    """Sample point pairs inside a radius-14 disc so rotation stays in bounds."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n_bits * 2:
        p = rng.integers(-15, 16, size=2)
        if p[0] ** 2 + p[1] ** 2 <= 14 ** 2:
            pts.append(p)
    pts = np.array(pts, dtype=np.float64)
    return pts[0::2], pts[1::2]


_P1, _P2 = _make_pattern()


def _smooth(img):
    ii = _integral(img.astype(np.float64))
    s = _box_sum(ii, 2) / 25.0
    return s  # valid-region smoothed image; s[i] centers on image coord (i + 2)


def describe(img, keypoints, angles):
    """Steered BRIEF: 256-bit descriptors, one uint8 array of 32 bytes per kp."""
    sm = _smooth(img)
    descs = np.zeros((len(keypoints), 32), dtype=np.uint8)
    for k, ((y, x), theta) in enumerate(zip(keypoints, angles)):
        c, s = math.cos(theta), math.sin(theta)
        rot = np.stack([_P1[:, 0] * c - _P1[:, 1] * s,
                        _P1[:, 0] * s + _P1[:, 1] * c], axis=1)
        rot2 = np.stack([_P2[:, 0] * c - _P2[:, 1] * s,
                         _P2[:, 0] * s + _P2[:, 1] * c], axis=1)
        p1 = np.rint(rot).astype(int) + (y - 2, x - 2)  # smooth idx = img - 2
        p2 = np.rint(rot2).astype(int) + (y - 2, x - 2)
        bits = sm[p1[:, 0], p1[:, 1]] < sm[p2[:, 0], p2[:, 1]]
        descs[k] = np.packbits(bits)
    return descs


def detect_and_describe(img, n=300, threshold=15):
    kps = detect_keypoints(img, n=n, threshold=threshold)
    if not kps:
        return np.zeros((0, 32), np.uint8), np.zeros((0, 2), int), np.zeros(0)
    angles = keypoint_orientations(img, kps)
    desc = describe(img, kps, angles)
    return desc, np.array(kps), angles


def _hamming_matrix(a, b):
    return np.bitwise_count(a[:, None, :] ^ b[None, :, :]).sum(-1)


def match(desc1, desc2, ratio=0.8, max_dist=64):
    """Mutual-nearest matching on Hamming distance with a ratio test."""
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    d = _hamming_matrix(desc1, desc2)
    nn1 = d.argmin(1)
    nn2 = d.argmin(0)
    out = []
    for i, j in enumerate(nn1):
        if nn2[j] != i:
            continue
        row = np.sort(d[i])
        if len(row) > 1 and row[1] > 0 and row[0] / row[1] > ratio:
            continue
        if d[i, j] <= max_dist:
            out.append((i, j, int(d[i, j])))
    return out
