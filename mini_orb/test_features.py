"""Offline tests for mini-ORB on synthetic images."""

import numpy as np
from scipy import ndimage

from mini_orb.features import (_smooth, detect_and_describe, detect_keypoints,
                               match)


def make_scene(size=400, seed=0):
    """Smooth random texture + dark frame: many distinct, repeatable corners."""
    rng = np.random.default_rng(seed)
    field = ndimage.gaussian_filter(rng.normal(0, 1, (size, size)), 3.0)
    field = (field - field.min()) / np.ptp(field) * 230 + 20
    img = field.copy()
    img[:20] = 0
    img[-20:] = 0
    img[:, :20] = 0
    img[:, -20:] = 0
    return img, None


def test_fast_finds_rect_corners():
    img, _ = make_scene()
    kps = detect_keypoints(img, n=200, threshold=20)
    assert len(kps) >= 150, len(kps)
    # inner frame corners (detection margin is 18px from image border)
    expected = [(21, 21), (21, 378), (378, 21), (378, 378)]
    for cy, cx in expected:
        d = min(abs(ky - cy) + abs(kx - cx) for ky, kx in kps)
        assert d <= 3, f"no keypoint near corner ({cy},{cx}), closest {d}px"


def test_transformed_image_matches():
    img, _ = make_scene()
    angle_deg, dy, dx = 10.0, 12, 9
    rotated = ndimage.rotate(img, angle_deg, reshape=False, order=1, mode="constant")
    shifted = ndimage.shift(rotated, (dy, dx), order=1, mode="constant")

    d1, k1, _ = detect_and_describe(img, n=400, threshold=20)
    d2, k2, _ = detect_and_describe(shifted, n=400, threshold=20)
    m = match(d1, d2, ratio=0.9, max_dist=90)
    assert len(m) >= 30, f"too few matches: {len(m)}"

    # verify transform consistency: rotated frame = R(angle) @ (p - c) + c + shift
    theta = np.deg2rad(-angle_deg)  # ndimage.rotate rotates the image content CCW,
    # which corresponds to rotating keypoint coordinates by -angle in image axes
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    c = np.array([199.5, 199.5])  # rotate keeps image center
    inliers = 0
    for i, j, _ in m:
        p1 = k1[i][::-1].astype(np.float64)
        p2 = k2[j][::-1].astype(np.float64)
        pred = R @ (p1 - c) + c + np.array([dx, dy])
        if np.linalg.norm(pred - p2) < 6.0:
            inliers += 1
    ratio = inliers / len(m)
    assert ratio > 0.5, f"transform inlier ratio too low: {ratio:.2f}"


def test_smooth_center_convention():
    # pin the index<->image-coordinate convention of the 5x5 box filter:
    # s[i] must center on image coordinate i + 2. Corner pixels are the only
    # unambiguous probe -- an off-by-one convention puts them in a different
    # window (this is the regression for the (-1, -1) BRIEF grid shift).
    img = np.zeros((40, 40))
    img[0, 0] = 255.0
    img[39, 39] = 255.0
    sm = _smooth(img)
    assert np.isclose(sm[0, 0], 255 / 25), sm[0, 0]    # window 0..4 sees (0,0)
    assert np.isclose(sm[35, 35], 255 / 25), sm[35, 35]  # window 35..39 sees (39,39)
    assert sm[2, 2] == 0.0                              # window 2..6 sees neither


def test_unrelated_images_match_poorly():
    a, _ = make_scene(seed=0)
    b = np.zeros((400, 400))
    ys, xs = np.mgrid[0:400, 0:400]
    b = ((np.sin(xs / 12.0) * np.cos(ys / 9.0) + 1) * 127).astype(float)
    da, _, _ = detect_and_describe(a, n=200, threshold=20)
    db, _, _ = detect_and_describe(b, n=200, threshold=20)
    m = match(da, db, ratio=0.8, max_dist=64)
    assert len(m) < 0.2 * max(len(da), 1), f"too many spurious matches: {len(m)}"


if __name__ == "__main__":
    test_fast_finds_rect_corners()
    print("FAST corner location test passed")
    test_transformed_image_matches()
    print("rotation+translation matching test passed")
    test_smooth_center_convention()
    print("smoothing center convention test passed")
    test_unrelated_images_match_poorly()
    print("unrelated-image rejection test passed")
    print("all mini-ORB tests passed")
