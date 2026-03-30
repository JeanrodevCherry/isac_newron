import os
import torch
import cv2
from src.isac_newron import (
    train_model,
    predict_and_crop,
    SmallUNet,
    show_prediction,
    parse_image,
)
import argparse
import sys
import glob
import pdb
from pathlib import Path

MAP_LOCATION = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {MAP_LOCATION}")


def main(
    model_path="pattern_model.pt",
    image_path="data/images/Plate_A_A2_Region1_Merged_ch00.jpg",
    model_type="cropper",
    train=False,
):
    # train_model("data/images", "data/masks", epochs=50)
    if train:
        if model_type == "cropper" and model_path == "pattern_model.pt":
            train_model("data/images", "data/masks", epochs=50)
        elif (
            model_type == "segmenter" and model_path == "segmentation_model.pt"
        ):
            train_model(
                "data/images", "data/masks", epochs=50, model_path=model_path
            )
    predict_one(
        model_path=model_path, image_path=image_path, model_type=model_type
    )


def predict_one(model_path, image_path, model_type="cropper"):
    # if not os.path.exists(model_path):
    #     train_model("data/images", "data/masks", epochs=50)
    # model = torch.load(model_path)

    model = SmallUNet()
    if not os.path.exists(model_path):
        # train_model("data/images", "data/masks", epochs=50)
        raise ValueError(
            f"Model not found at {model_path}. Training completed, but model file is missing."
        )
    if model_type == "cropper":
        checkpoint = torch.load(model_path, map_location=MAP_LOCATION)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif model_type == "segmenter":
        model.load_state_dict(torch.load(model_path, map_location=MAP_LOCATION))
    model.eval()

    # crop = predict_and_crop(model, image_path)
    # label = show_prediction(model,image_path)
    label, mask = parse_image(model, image_path, model_type=model_type)
    output_path = Path(image_path).with_suffix(".tiff")
    # cv2.imwrite("cropped_result.jpg", crop)
    cv2.imwrite(output_path, label)


if __name__ == "__main__":
    # model_path = "pattern_model.pt"
    parser = argparse.ArgumentParser("Crazy Cropper with its two newrons")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model (default: False)",
        default=False,
    )
    # parser.add_mutually_exclusive_group(required=False)
    parser.add_argument(
        "--filename",
        default="./data/images/data/images/D03_A1_Region1_ch00.jpg",
        required=False,
    )
    parser.add_argument("--directory", default=None, type=Path, required=False)
    parser.add_argument(
        "--model-type",
        default="cropper",
        type=str,
        required=False,
        choices=["cropper", "segmenter"],
        help="Type of the model to use (default: cropper)",
    )
    args = parser.parse_args()
    # if args.filename:
    #     main(model_path, args.filename)
    model_type = args.model_type
    model_path = (
        "pattern_model.pt"
        if model_type == "cropper"
        else "segmentation_model.pt"
    )
    # if args.train:
    #     train = args.train

    pdb.set_trace()
    if args.directory:
        # for image in glob.glob(args.directory):
        #     print(image)
        for image in sorted(glob.glob(os.path.join(args.directory, "*.jpg"))):
            print(image)
            # pdb.set_trace()
            main(model_path, image, model_type, args.train)
