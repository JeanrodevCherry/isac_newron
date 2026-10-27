"""Auto-label spheroids of data/noise_filter images and refine pattern masks.

Outputs (QA panel in debug_out/label_sheet.jpg):
  data/noise_filter/masks_refined/<stem>.tif   pattern masks (lattice hull, 1/8 resolution)
  data/segmentation/images/<stem>.jpg + .json  new spheroid training samples (labelme polygons)
"""
import glob, json, os, pickle, shutil, sys
from pathlib import Path
import cv2, numpy as np, tifffile

from src.labeling import detect_spheroids, lattice_pattern_mask

NF = Path("data/noise_filter")
SEG = Path("data/segmentation/images")
CACHE = Path("/tmp/label_cache.pkl")
MIN_SPHEROIDS, MAX_SPHEROIDS = 30, 70


def mask_path(stem):
    return glob.glob(str(NF / "masks" / (glob.escape(stem) + "_label.tif*")))[0]


def coarse_box(stem, shape):
    ys, xs = np.where(tifffile.imread(mask_path(stem)) > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(0.15 * (x1 - x0))
    return (max(0, x0 - pad), max(0, y0 - pad), min(shape[1], x1 + pad), min(shape[0], y1 + pad))


def labelme(stem, shape, contours, path):
    data = {"version": "7.9.1", "flags": {}, "imagePath": stem + ".jpg", "imageData": None,
            "imageHeight": shape[0], "imageWidth": shape[1],
            "shapes": [{"label": "1", "points": c.tolist(), "group_id": None,
                        "description": "", "shape_type": "polygon", "flags": {}} for c in contours]}
    path.write_text(json.dumps(data))


def run():
    res = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    for p in sorted(glob.glob(str(NF / "images" / "*.jpg"))):
        stem = Path(p).stem
        if stem.startswith("Plate_3_") or stem in res:
            continue
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        box = coarse_box(stem, img.shape)
        r = detect_spheroids(img, box)
        r["shape"] = img.shape
        r["small"] = r["small"].copy()
        res[stem] = r
        print(stem, len(r["contours"]), flush=True)
        pickle.dump(res, open(CACHE, "wb"))
    return res


def sheet(res):
    tiles = []
    for stem, r in res.items():
        v = cv2.cvtColor(cv2.createCLAHE(3, (8, 8)).apply(r["small"]), cv2.COLOR_GRAY2BGR)
        off, s = np.array(r["box"][:2]), r["s"]
        for c in r["contours"]:
            cv2.polylines(v, [((c - off) * s).astype(np.int32)], True, (0, 0, 255), 2)
        if len(r["centers"]):
            pm = lattice_pattern_mask((r["centers"] - off) * s, r["pitch"] * s, v.shape[:2])
            cs, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(v, cs, -1, (0, 255, 0), 2)
        v = cv2.resize(v, (330, 330))
        cv2.putText(v, f"{stem[:26]} n={len(r['contours'])}", (2, 12), 0, 0.4, (0, 255, 255), 1)
        tiles.append(v)
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    Path("debug_out").mkdir(exist_ok=True)
    cv2.imwrite("debug_out/label_sheet.jpg", np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]))


def plate3_pattern_mask(img, scale=8):
    """Plate_3 images are already denoised: the pattern is the non-black region."""
    small = cv2.resize(img, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
    binary = (cv2.GaussianBlur(small, (0, 0), 2) > 3).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary)
    out = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    cs, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out[:] = 0
    cv2.drawContours(out, cs, -1, 1, -1)
    return out


def write_outputs(res, scale=8):
    (NF / "masks_refined").mkdir(exist_ok=True)
    SEG.mkdir(parents=True, exist_ok=True)
    accepted, rejected = [], []
    for stem, r in res.items():
        n = len(r["contours"])
        if not MIN_SPHEROIDS + 20 <= n <= MAX_SPHEROIDS - 12:  # 50..58 wells expected (55 nominal)
            rejected.append((stem, n))
            continue
        shape = r["shape"]
        small_shape = (shape[0] // scale, shape[1] // scale)
        pm = lattice_pattern_mask(r["centers"] / scale, r["pitch"] / scale, small_shape)
        tifffile.imwrite(NF / "masks_refined" / f"{stem}_label.tif", pm, compression="zlib")
        shutil.copy(NF / "images" / f"{stem}.jpg", SEG / f"{stem}.jpg")
        labelme(stem, shape, r["contours"], SEG / f"{stem}.json")
        accepted.append((stem, n))

    for p in sorted(glob.glob(str(NF / "images" / "Plate_3_*.jpg"))):
        stem = Path(p).stem
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        tifffile.imwrite(NF / "masks_refined" / f"{stem}_label.tif", plate3_pattern_mask(img, scale),
                         compression="zlib")
        dots = (tifffile.imread(mask_path(stem)) > 0).astype(np.uint8)
        cs, _ = cv2.findContours(dots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = [cv2.approxPolyDP(c, 1.0, True)[:, 0, :].astype(float) for c in cs if cv2.contourArea(c) > 50]
        shutil.copy(p, SEG / f"{stem}.jpg")
        labelme(stem, img.shape, polys, SEG / f"{stem}.json")
        accepted.append((stem, len(polys)))
    print("accepted", len(accepted), accepted)
    print("rejected", rejected)


if __name__ == "__main__":
    res = run()
    sheet(res)
    if "--write" in sys.argv:
        write_outputs(res)
