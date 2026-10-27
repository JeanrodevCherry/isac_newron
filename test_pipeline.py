"""End-to-end test of the two-model pipeline with visual checkpoints.

Stage 1  pattern model   -> IoU / border F-score vs data/noise_filter/masks_refined
                            panels: debug_out/test/pattern_<stem>.jpg
Stage 2  denoise + crop  -> pattern region must be found, crop must contain every labelled spheroid
Stage 3  spheroid model  -> IoU / Dice vs labelme polygons in data/segmentation/images
                            panels: debug_out/test/spheroid_<stem>.jpg

Usage:
    python test_pipeline.py                       # all images
    python test_pipeline.py --holdout A B C       # only report these stems in the summary
Exit code is 1 if a mean metric is below its threshold.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from src.datasets import PatternDataset, SpheroidDataset
from src.models import load_pattern_model, load_spheroid_model
from src.pattern import evaluate_pattern_model
from src.segmentation import predict_spheroid_prob
from src.toolkit import pattern_region
from validate_pipeline import tile

NOISE_IMG, NOISE_MASK = "data/noise_filter/images", "data/noise_filter/masks_refined"
SEG_DIR = "data/segmentation/images"


def stage1(pm, out, only, thr):
    ds = PatternDataset(NOISE_IMG, NOISE_MASK, size=pm.input_size, names=only)
    iou, bf = evaluate_pattern_model(pm, ds)
    print(f"[1] pattern model ({len(ds)} images): IoU={iou:.3f} borderF={bf:.3f}")
    for name, (img, gt) in zip(ds.names, ds.cache):
        with torch.no_grad():
            prob = torch.sigmoid(pm(torch.from_numpy(img).float()[None, None] / 255.0))[0, 0].numpy()
        ov = cv2.cvtColor(np.clip(img.astype(np.float32) * 120 / max(img.mean(), 1), 0, 255).astype(np.uint8),
                          cv2.COLOR_GRAY2BGR)
        for m, c in (((gt > 0.5), (0, 255, 0)), ((prob > 0.5), (0, 0, 255))):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, cs, -1, c, 1)
        panel = np.hstack([tile(img, "input"), tile((gt * 255).astype(np.uint8), "GT pattern"),
                           tile((prob * 255).astype(np.uint8), "pred prob"), tile(ov, "pred red / GT green")])
        cv2.imwrite(str(out / f"pattern_{name}.jpg"), panel)
    return iou >= thr


def stage23(pm, sm, out, only, thr):
    helper = SpheroidDataset(SEG_DIR, SEG_DIR)
    rows, crop_ok = [], True
    for p in sorted(Path(SEG_DIR).glob("*.jpg")):
        if only and p.stem.replace(" ", "_") not in only and p.stem not in only:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        pmask, box, pprob = pattern_region(pm, img)
        if pmask is None:
            print(f"[2] {p.name}: NO PATTERN")
            crop_ok = False
            continue
        x1, y1, x2, y2 = box
        gt_full = helper._load_spheroid_mask(str(p.with_suffix(".json")), img.shape) > 0
        inside = gt_full[y1:y2, x1:x2].sum() / max(gt_full.sum(), 1)
        crop_ok &= inside > 0.99
        denoised = (img * pmask)[y1:y2, x1:x2]
        gt = gt_full[y1:y2, x1:x2]
        prob = predict_spheroid_prob(sm, denoised, pmask[y1:y2, x1:x2])
        pred = prob > 0.5
        i = (pred & gt).sum()
        iou, dice = i / max((pred | gt).sum(), 1), 2 * i / max(pred.sum() + gt.sum(), 1)
        n_gt = cv2.connectedComponents(gt.astype(np.uint8))[0] - 1
        n_pr = cv2.connectedComponents(pred.astype(np.uint8))[0] - 1
        rows.append((p.name, iou, dice, inside, n_gt, n_pr))

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
        cv2.imwrite(str(out / f"spheroid_{p.stem}.jpg"), panel)
        print(f"    {p.name}: crop_has_GT={inside:.3f} IoU={iou:.3f} Dice={dice:.3f} "
              f"spheroids GT/pred={n_gt}/{n_pr}")
    if not rows:
        print("[2/3] no images evaluated")
        return False
    miou = float(np.mean([r[1] for r in rows]))
    mdice = float(np.mean([r[2] for r in rows]))
    print(f"[2] crop contains all spheroids: {'OK' if crop_ok else 'FAIL'}")
    print(f"[3] spheroid model ({len(rows)} images): mean IoU={miou:.3f} mean Dice={mdice:.3f}")
    return crop_ok and miou >= thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="pattern_model.pt")
    ap.add_argument("--model", default="segmentation_model.pt")
    ap.add_argument("--out", default="debug_out/test")
    ap.add_argument("--holdout", nargs="*", help="only evaluate these image stems")
    ap.add_argument("--pattern-iou", type=float, default=0.9)
    ap.add_argument("--spheroid-iou", type=float, default=0.6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {s.replace(" ", "_") for s in a.holdout} if a.holdout else None

    pm, _, _ = load_pattern_model(a.pattern)
    sm, _, _ = load_spheroid_model(a.model)
    ok1 = stage1(pm, out, only, a.pattern_iou)
    ok2 = stage23(pm, sm, out, only, a.spheroid_iou)
    print("PASS" if ok1 and ok2 else "FAIL", f"- panels in {out}/")
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
