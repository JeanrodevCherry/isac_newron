from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import tifffile
import torch
from PIL import Image
from cellpose import models
import cv2

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create binary Cellpose-SAM masks for JPG images."
    )
    parser.add_argument("input_folder", type=Path)
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Keep detected cells with at least this many pixels (default: 500).",
    )
    args = parser.parse_args()

    if not args.input_folder.is_dir():
        parser.error(f"Not a folder: {args.input_folder}")
    if args.min_area < 1:
        parser.error("--min-area must be at least 1.")

    # Cellpose 4.2+ includes cpsam_v2. For an older install, use "cpsam".
    #model = models.CellposeModel(pretrained_model="cpsam_v2")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check your NVIDIA driver and PyTorch installation."
        )

    device = torch.device("cuda:0")
    model = models.CellposeModel(
        pretrained_model="cpsam_v2",
        device=device,
    )
    image_paths = sorted(
        path
        for path in args.input_folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )
    if not image_paths:
        print(f"No JPG images found in {args.input_folder}")
        return
    print(f"Cellpose-SAM using: {model.device}")
    for image_path in tqdm(image_paths,desc="Segmenting Images",unit="image"):
        with Image.open(image_path) as image:
            image_array = np.asarray(image.convert("RGB"))
            h, w = image_array.shape[:2]
            scale = min(1.0, 1000 / max(h, w))
            new_size = (round(w * scale), round(h * scale))
            small_image = cv2.resize(image_array, new_size, interpolation=cv2.INTER_AREA)
            labels,*_ = model.eval(small_image,flow_threshold=0.4,
        cellprob_threshold=0.4,
        niter=250)
        #labels, *_ = model.eval(image_array)

        # Keep only detected objects meeting the minimum area.
        areas = np.bincount(labels.ravel())
        keep = areas >= args.min_area
        keep[0] = False  # Label 0 is background.
        binary_mask = keep[labels].astype(np.uint8)

        output_path = image_path.with_name(
            f"{image_path.stem}_labeled.tif"
        )
        tifffile.imwrite(output_path, binary_mask)

        print(f"Saved {output_path.name}")

    print(f"Processed {len(image_paths)} image(s).")


if __name__ == "__main__":
    main()
