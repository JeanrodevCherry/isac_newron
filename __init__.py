import os
import torch
import cv2
import numpy as np
# from src.isac_newron import train_model, predict_and_crop, SmallUNet, show_prediction, parse_image, extract_state_dict, train_segmentation_model, segment_image

import argparse
import sys
from pathlib import Path
import glob
# from glob import glob

from src.segmentation import train_segmentation_model, segment_image
from src.toolkit import build_overlay

def main(model_path="pattern_model.pt",image_path="data/images/Plate_A_A2_Region1_Merged_ch00.jpg"):
    # train_model("data/images", "data/masks", epochs=100)
    predict_one(model_path=model_path,image_path=image_path)



def resize_for_display(image, max_width=1200, max_height=1000):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale == 1.0:
        return image

    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def predict_one(model_path, image_path, output_path=None):
    """Run the pattern model on one image and save crop, mask and overlay."""
    from src.models import load_pattern_model
    from src.toolkit import pattern_region

    model, loaded, _ = load_pattern_model(model_path)
    quality = loaded.get("training_quality", {}) if isinstance(loaded, dict) else {}
    if quality:
        print(f"📊 Trained model quality -> final: {quality.get('final_train_loss')}, "
              f"best: {quality.get('best_train_loss')} (epoch {quality.get('best_epoch')})")

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    pmask, box, _ = pattern_region(model, img)
    if pmask is None:
        print("⚠️ No pattern detected.")
        return
    x1, y1, x2, y2 = box
    crop = (img * pmask)[y1:y2, x1:x2]
    label = pmask[y1:y2, x1:x2].astype(np.float32)
    overlay, _ = build_overlay(crop, label)
    if not output_path:
        path_ = Path(image_path).resolve()
        output_path = path_.with_stem(path_.stem + "_pattern_mask").with_suffix(".tif")
    print(output_path)
    cv2.imwrite("cropped_result.jpg", crop)
    cv2.imwrite(str(output_path), (label * 255).astype(np.uint8))
    cv2.imwrite("overlay_result.jpg", overlay)


if __name__=="__main__":
    model_path = "pattern_model.pt"
    segmentation_model_path = "segmentation_model.pt"
    parser = argparse.ArgumentParser("Crazy Cropper with its two newrons")
    parser.add_argument("-filename",default="./data/segmentation/images/Plate_1_D26_A1 Region1_Merged_ch00.jpg",required=False)
    parser.add_argument("--train",action="store_true",help="Train segmentation model")
    parser.add_argument("--segment",action="store_true",help="Segment")
    args = parser.parse_args()

    # if args.filename:
    #     main(model_path,args.filename)
    # if args.train:
    #     train_segmentation_model(model_path=segmentation_model_path)
    
    if args.train:
        segmentation_dir = "data/segmentation/images/"
        train_segmentation_model(segmentation_dir, segmentation_dir,
                                 model_path=segmentation_model_path)
    if args.segment:
        print(args.filename)
        segment_image(segmentation_model_path, Path(args.filename))
    if not (args.train or args.segment):
        predict_one(model_path, args.filename)
