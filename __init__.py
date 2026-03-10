import os
import torch
import cv2
from src.isac_newron import train_model, predict_and_crop, SmallUNet, show_prediction, parse_image, extract_state_dict
import argparse
import sys

def main(model_path="pattern_model.pt",image_path="data/images/Plate_A_A2_Region1_Merged_ch00.jpg"):
    # train_model("data/images", "data/masks", epochs=100)
    predict_one(model_path=model_path,image_path=image_path)

def predict_one(model_path,image_path):
    # if not os.path.exists(model_path):
    #     train_model("data/images", "data/masks", epochs=50)
    # model = torch.load(model_path)

    model = SmallUNet()
    if not os.path.exists(model_path):
        train_model("data/images", "data/masks", epochs=50)

    loaded = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(extract_state_dict(loaded))

    if isinstance(loaded, dict) and "training_quality" in loaded:
        quality = loaded["training_quality"]
        final_loss = quality.get("final_train_loss")
        best_loss = quality.get("best_train_loss")
        best_epoch = quality.get("best_epoch")
        print(f"📊 Trained model quality -> final: {final_loss}, best: {best_loss} (epoch {best_epoch})")

    model.eval()

    # crop = predict_and_crop(model, image_path)
    # label = show_prediction(model,image_path)
    crop,label = parse_image(model,image_path)
    cv2.imwrite("cropped_result.jpg", crop)
    cv2.imwrite("mask.tif", label)

if __name__=="__main__":
    model_path = "pattern_model.pt"
    parser = argparse.ArgumentParser("Crazy Cropper with its two newrons")
    parser.add_argument("-filename",default="./data/images/data/images/D03_A1_Region1_ch00.jpg",required=False)
    args = parser.parse_args()

    if args.filename:
        main(model_path,args.filename)
