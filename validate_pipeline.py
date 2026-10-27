"""Step-by-step visual validation: pattern model (denoise) -> spheroid segmentation.

Writes one panel per image to debug_out/ and prints IoU/Dice against the labelme GT:
1 original + crop box | 2 pattern prob | 3 largest pattern region | 4 denoised crop
| 5 GT spheroids | 6 spheroid prob | 7 overlay (pred red / GT green)
"""
import argparse
from pathlib import Path
import cv2, numpy as np, torch

from src.datasets import SpheroidDataset
from src.models import load_pattern_model, load_spheroid_model
from src.segmentation import SPHEROID_SIZE, predict_spheroid_prob
from src.toolkit import pattern_region


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
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    pm, _, _ = load_pattern_model(a.pattern)
    sm, _, _ = load_spheroid_model(a.model)
    helper = SpheroidDataset(a.images, a.images)  # only used to rasterize labelme polygons
    for p in sorted(Path(a.images).glob("*.jpg")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        pmask, box, pprob = pattern_region(pm, img)
        if pmask is None:
            print(f"{p.name}: NO PATTERN"); continue
        x1, y1, x2, y2 = box
        denoised = (img * pmask)[y1:y2, x1:x2]
        gt = helper._load_spheroid_mask(str(p.with_suffix(".json")), img.shape)[y1:y2, x1:x2] > 0
        prob = predict_spheroid_prob(sm, denoised, pmask[y1:y2, x1:x2])
        pred = prob > a.threshold

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


if __name__ == "__main__":
    main()
