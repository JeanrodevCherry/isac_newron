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

from src.isac_newron import predict_and_crop, parse_image
from src.segmentation import train_segmentation_model, segment_image
from src.models import SmallUNet, extract_state_dict
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

def predict_one(model_path,image_path, output_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallUNet().to(device)
    if not os.path.exists(model_path):
        train_model("data/images", "data/masks", epochs=50)

    loaded = torch.load(model_path, map_location=device)
    model.load_state_dict(extract_state_dict(loaded))

    if isinstance(loaded, dict) and "training_quality" in loaded:
        quality = loaded["training_quality"]
        final_loss = quality.get("final_train_loss")
        best_loss = quality.get("best_train_loss")
        best_epoch = quality.get("best_epoch")
        print(f"📊 Trained model quality -> final: {final_loss}, best: {best_loss} (epoch {best_epoch})")

    model.eval()

    parsed = parse_image(model, image_path)
    if isinstance(parsed, tuple):
        crop, label = parsed
    else:
        crop = parsed
        label = np.zeros_like(crop, dtype=np.float32)

    overlay, mask_vis = build_overlay(crop, label)
    if not output_path:
        path_ = Path(image_path).resolve()
        output_path = path_.with_stem(path_.stem + "_pattern_mask").with_suffix(".tif")
    # else:
    #     output_path = "mask.tif"
    print(output_path)
    cv2.imwrite("cropped_result.jpg", crop)
    cv2.imwrite(output_path, label)
    cv2.imwrite("overlay_result.jpg", overlay)

    # cv2.imshow("Output", resize_for_display(crop))
    # cv2.imshow("Mask Overlay", resize_for_display(overlay, 1299, 1000))
    # cv2.imshow("Mask", resize_for_display(mask_vis))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


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
    
    import pdb
    
    if args.segment or args.train:
        segmentation_dir = "data/segmentation/images/"
        segmentation_mask_dir =  "data/segmentation/masks/"
        for file in sorted(glob.glob(os.path.join(segmentation_dir, "*.jpg"))):
            stem = Path(file).stem
            filename = (Path(segmentation_mask_dir) / stem).with_suffix(".tif")
            predict_one(model_path=model_path,image_path=file,output_path=filename)
        if args.train:
            model = train_segmentation_model(segmentation_dir,segmentation_dir,segmentation_mask_dir,model_path=segmentation_model_path)
        if args.segment:
            print(args.filename)
            segment_image(segmentation_model_path,Path(args.filename))