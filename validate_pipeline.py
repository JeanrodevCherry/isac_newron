"""Step-by-step visual validation: pattern model (denoise) -> spheroid segmentation.

Writes one panel per image to debug_out/ and prints IoU/Dice against the labelme GT:
1 original + crop box | 2 pattern prob | 3 largest pattern region | 4 denoised crop
| 5 GT spheroids | 6 spheroid prob | 7 overlay (pred red / GT green)
"""
import argparse
import re
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch

from src.datasets import SpheroidDataset
from src.models import load_pattern_model, load_spheroid_model
from src.segmentation import SPHEROID_SIZE, predict_spheroid_prob, remove_close_spheroids, remove_holes
from src.toolkit import pattern_region, format_day


def parse_name(name):
    g = lambda pat: (m.group(1) if (m := re.search(pat, name)) else "")
    return g(r"Plate_(\d+)"), g(r"(?<![A-Za-z0-9])(D\d{2})(?![0-9])"), g(r"(?<![A-Za-z0-9])([A-D][1-6])(?![0-9])")


def label_rows(labels, image_number, path, um_per_px):
    plate, day, well = parse_name(path.name)
    # print(um_per_px)
    rows = []
    for k, l in enumerate(np.unique(labels[labels > 0]), start=1):
        area = (labels == l).sum() * um_per_px ** 2
        rows.append(dict(label=k, ImageNumber=image_number, AreaShape_Area=area,
                         AreaShape_EquivalentDiameter=np.sqrt(4 * area / np.pi),
                         ImagePath=str(path.resolve()), Day=format_day(day), Well=well,
                         Tile="", z="", Condition=f"Plate_{plate}" if plate else ""))
    return rows


def tile(img, title, w=380):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (w, w))
    cv2.putText(img, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/segmentation/images")
    ap.add_argument("--pattern", default="pattern_model.pt")
    ap.add_argument("--model", default="segmentation_model.pt")
    ap.add_argument("--out", default="debug_out")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--pattern-um", type=float, default=600, help="physical pattern width in um")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    pm, _, _ = load_pattern_model(a.pattern)
    sm, _, _ = load_spheroid_model(a.model)
    helper = SpheroidDataset(a.images, a.images)  # only used to rasterize labelme polygons
    rows = []
    for n, p in enumerate(sorted(Path(a.images).glob("*.jpg"))):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        pmask, box, pprob = pattern_region(pm, img)
        if pmask is None:
            print(f"{p.name}: NO PATTERN"); continue
        x1, y1, x2, y2 = box
        denoised = (img * pmask)[y1:y2, x1:x2]
        gt = helper._load_spheroid_mask(str(p.with_suffix(".json")), img.shape)[y1:y2, x1:x2] > 0
        prob = predict_spheroid_prob(sm, denoised, pmask[y1:y2, x1:x2])
        pred = prob > a.threshold
        labels = remove_close_spheroids(remove_holes(pred), denoised)
        rows += label_rows(labels, n, p, a.pattern_um * 3 / img.shape[1])  # pattern width ~ 1/3 of the image width

        full = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(full, (x1, y1), (x2, y2), (0, 0, 255), 40)
        ov = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        for m, c in ((gt, (0, 255, 0)), (pred, (0, 0, 255))):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, cs, -1, c, 3)
        panel = np.hstack([tile(full, "1 original + crop"),
                           tile((pprob * 255).astype(np.uint8), "2 pattern prob"),
                           tile(pmask * 255, "3 largest region"),
                           tile(denoised, "4 denoised crop"),
                           tile(gt.astype(np.uint8) * 255, "5 GT spheroids"),
                           tile((prob * 255).astype(np.uint8), "6 spheroid prob"),
                           tile(ov, "7 pred red / GT green")])
        cv2.imwrite(str(out / f"{p.stem}.jpg"), panel)
        i = (pred & gt).sum()
        print(f"{p.name}: pattern_cover={pmask.mean():.3f} crop={x2-x1}x{y2-y1} "
              f"IoU={i/max((pred | gt).sum(),1):.3f} Dice={2*i/max(pred.sum()+gt.sum(),1):.3f}")
    with pd.ExcelWriter( out /"results.xlsx") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False)


if __name__ == "__main__":
    main()
