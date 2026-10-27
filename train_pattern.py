"""Train the PatternUNet. Usage: python train_pattern.py [--val] [--epochs N] [--out path]"""
import argparse
import torch
from src.pattern import train_pattern_model, evaluate_pattern_model
from src.datasets import PatternDataset
from src.models import load_pattern_model

ap = argparse.ArgumentParser()
ap.add_argument("--val", action="store_true", help="hold out 5 images and compare with the current model")
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--size", type=int, default=384)
ap.add_argument("--out", default="pattern_model_v2.pt")
a = ap.parse_args()
torch.set_num_threads(8)
IMG, MASK = "data/noise_filter/images", "data/noise_filter/masks_refined"
val = ["D24_B2_Region1_Merged_ch00", "Plate_A_A3_Region1_Merged_ch01", "Plate_B_A1_Region1_Merged_ch00",
       "Plate_E_A2_Region1_Merged_ch00", "Plate_3_D24_B5_Region1_Merged_ch00"] if a.val else None
train_pattern_model(IMG, MASK, epochs=a.epochs, size=a.size, model_path=a.out, val_names=val)
if val:
    old, _, _ = load_pattern_model("pattern_model.pt")
    new, _, _ = load_pattern_model(a.out)
    for name, m in (("old", old), ("new", new)):
        ds = PatternDataset(IMG, MASK, size=m.input_size, names=val)
        print(name, "val IoU=%.4f borderF=%.4f" % evaluate_pattern_model(m, ds))
