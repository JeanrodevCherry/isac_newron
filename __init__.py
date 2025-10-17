import os
import torch
import cv2
from src.isac_newron import train_model, predict_and_crop, SmallUNet, show_prediction
import argparse
import sys

def main(model_path="pattern_model.pt",image_path="data/images/Plate_A_A2_Region1_Merged_ch00.jpg"):
    # train_model("data/images", "data/masks", epochs=50)
    predict_one(model_path=model_path,image_path=image_path)

def predict_one(model_path,image_path):
    # if not os.path.exists(model_path):
    #     train_model("data/images", "data/masks", epochs=50)
    # model = torch.load(model_path)

    model = SmallUNet()
    if not os.path.exists(model_path):
        train_model("data/images", "data/masks", epochs=50)
    model.load_state_dict(torch.load("pattern_model.pt", map_location="cpu"))
    model.eval()

    crop = predict_and_crop(model, image_path)
    label = show_prediction(model,image_path)
    cv2.imwrite("cropped_result.jpg", crop)
    cv2.imwrite("mask.tif", label)
if __name__=="__main__":
    model_path = "pattern_model.pt"
    parser = argparse.ArgumentParser("Crazy Cropper with its two newrons")
    parser.add_argument("-filename",default="data/images/Plate_A_A2_Region1_Merged_ch00.jpg",required=False)
    args = parser.parse_args()

    if args.filename:
        main(model_path,args.filename)
    #     print("ok")
    #     main()
