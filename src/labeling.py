"""Classical lattice-based detector used to auto-label spheroids and refine pattern masks."""
import cv2
import numpy as np

WORK = 1200  # working width (px) of the pattern crop


def _prep(img, box):
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    s = WORK / max(crop.shape)
    small = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return small, s


def _largest_hex_component(c, dm, pitch, ang_tol=np.deg2rad(12)):
    """Indices of the largest set of points linked by hexagonal-lattice bonds."""
    i, j = np.where(np.triu((dm > 0.85 * pitch) & (dm < 1.15 * pitch)))
    if len(i) == 0:
        return []
    ang = np.arctan2(c[j, 1] - c[i, 1], c[j, 0] - c[i, 0])
    theta = np.angle(np.exp(6j * ang).mean()) / 6
    dev = np.abs(np.angle(np.exp(6j * (ang - theta)))) / 6
    i, j = i[dev < ang_tol], j[dev < ang_tol]
    while len(i):  # prune stray points hanging on a single bond
        deg = np.bincount(np.r_[i, j], minlength=len(c))
        keep = (deg[i] >= 2) & (deg[j] >= 2)
        if keep.all():
            break
        i, j = i[keep], j[keep]
    if len(i) == 0:
        return []
    parent = list(range(len(c)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(i, j):
        parent[find(a)] = find(b)
    roots = np.array([find(k) for k in range(len(c))])
    vals, counts = np.unique(roots[np.unique(np.r_[i, j])], return_counts=True)
    return list(np.where(roots == vals[np.argmax(counts)])[0])


def detect_spheroids(img, box, grow=1.12):
    """Find spheroids lying on the hexagonal micro-well lattice inside `box` (x1,y1,x2,y2).

    Returns dict with: contours (full-res polygons), centers (full-res xy),
    pitch (full-res px), small/s/box for visualisation.
    """
    small, s = _prep(img, box)
    f = small.astype(np.float32)
    bg = cv2.GaussianBlur(f, (0, 0), 35)
    d = np.clip(bg - cv2.GaussianBlur(f, (0, 0), 1.5), 0, 255)
    d8 = np.clip(d * (255.0 / max(d.max(), 1)), 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (d8 > thr).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary)
    out = dict(contours=[], centers=np.zeros((0, 2)), pitch=0.0, small=small, s=s, box=box)
    if n < 8:
        return out
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    med = np.median(areas)
    keep = [i + 1 for i, a in enumerate(areas) if 0.25 * med <= a <= 3.0 * med]
    c = cents[keep]
    if len(c) < 8:
        return out
    dm = np.linalg.norm(c[:, None] - c[None], axis=2)
    np.fill_diagonal(dm, np.inf)
    pitch = float(np.median(dm.min(1)))
    for _ in range(3):  # refine the pitch using lattice-like pairs only
        pairs = dm[(dm > 0.8 * pitch) & (dm < 1.2 * pitch)]
        if len(pairs) == 0:
            return out
        pitch = float(np.median(pairs))
    good = [keep[i] for i in _largest_hex_component(c, dm, pitch)]
    if len(good) < 8:
        return out
    contours = []
    centers = []
    for k in good:
        m = (labels == k).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cs, key=cv2.contourArea)
        cnt = cv2.approxPolyDP(cnt, 0.8, True)[:, 0, :].astype(np.float64)
        if len(cnt) < 3:
            continue
        cnt = cents[k] + (cnt - cents[k]) * grow  # thresholding under-segments the dark core edge
        contours.append(cnt / s + np.array([box[0], box[1]]))
        centers.append(cents[k] / s + np.array([box[0], box[1]]))
    out.update(contours=contours, centers=np.array(centers), pitch=pitch / s)
    return out


def lattice_pattern_mask(centers, pitch, shape, radius_ratio=0.55):
    """Convex hull of one disc per well -> solid pattern region hugging the outer wells."""
    mask = np.zeros(shape, np.uint8)
    r = int(radius_ratio * pitch)
    for cx, cy in centers:
        cv2.circle(mask, (int(cx), int(cy)), r, 1, -1)
    pts = cv2.findNonZero(mask)
    hull = cv2.convexHull(pts)
    out = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(out, hull, 1)
    return out
