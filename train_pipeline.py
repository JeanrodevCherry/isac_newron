"""Train either model in the image-processing pipeline."""
import argparse

import torch

from src.datasets import PatternDataset
from src.models import load_pattern_model
from src.pattern import evaluate_pattern_model, train_pattern_model
from src.segmentation import train_segmentation_model


def build_parser():
    parser = argparse.ArgumentParser(description="Train a model in the image-processing pipeline.")
    parser.add_argument(
        "--model",
        choices=("pattern", "segmentation"),
        required=True,
        help="which model to train",
    )
    parser.add_argument("--val", action="store_true", help="hold out 5 images to compare pattern models")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--size", type=int, default=384, help="pattern model input size")
    parser.add_argument("--batch-size", type=int, help="training batch size")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--out", help="checkpoint path (defaults to the selected model's standard path)")
    parser.add_argument("--pattern-img-dir", default="data/noise_filter/images")
    parser.add_argument("--pattern-mask-dir", default="data/noise_filter/masks_refined")
    parser.add_argument("--segmentation-img-dir", default="data/segmentation/images")
    parser.add_argument("--segmentation-json-dir", default="data/segmentation/images")
    parser.add_argument("--segmentation-mask-dir", help="optional directory of honeycomb TIF masks")
    parser.add_argument("--pattern-model", default="pattern_model.pt",
                        help="pattern checkpoint used to prepare segmentation training samples")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model == "segmentation" and args.val:
        parser.error("--val is only supported when training the pattern model")

    torch.set_num_threads(8)
    if args.model == "pattern":
        batch_size = args.batch_size if args.batch_size is not None else 4
        lr = args.lr if args.lr is not None else 2e-3
        model_path = args.out or "pattern_model_v2.pt"
        val_names = [
            "D24_B2_Region1_Merged_ch00",
            "Plate_A_A3_Region1_Merged_ch01",
            "Plate_B_A1_Region1_Merged_ch00",
            "Plate_E_A2_Region1_Merged_ch00",
            "Plate_3_D24_B5_Region1_Merged_ch00",
        ] if args.val else None
        train_pattern_model(
            args.pattern_img_dir,
            args.pattern_mask_dir,
            epochs=args.epochs,
            lr=lr,
            batch_size=batch_size,
            size=args.size,
            model_path=model_path,
            val_names=val_names,
        )
        if val_names:
            old, _, _ = load_pattern_model("pattern_model.pt")
            new, _, _ = load_pattern_model(model_path)
            for name, model in (("old", old), ("new", new)):
                dataset = PatternDataset(
                    args.pattern_img_dir,
                    args.pattern_mask_dir,
                    size=model.input_size,
                    names=val_names,
                )
                print(name, "val IoU=%.4f borderF=%.4f" % evaluate_pattern_model(model, dataset))
        return

    train_segmentation_model(
        args.segmentation_img_dir,
        args.segmentation_json_dir,
        mask_dir=args.segmentation_mask_dir,
        epochs=args.epochs,
        lr=args.lr if args.lr is not None else 1e-3,
        batch_size=args.batch_size if args.batch_size is not None else 2,
        model_path=args.out or "segmentation_model.pt",
        pattern_model_path=args.pattern_model,
    )


if __name__ == "__main__":
    main()
