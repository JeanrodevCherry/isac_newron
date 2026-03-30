import cv2
import glob
from pathlib import Path
import os

_JPEG_QUALITY = 100


def _to_jpg(input_filename: Path):
    if isinstance(input_filename, str):  # conversion
        input_filename = Path(input_filename)
    if input_filename.suffix == ".jpg":
        return input_filename

    jpg_path = input_filename.with_suffix(".jpg")
    image = cv2.imread(str(input_filename), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(
            f"Could not convert image: {str(input_filename)} to jpg"
        )
    ok = cv2.imwrite(
        str(jpg_path), image, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError(f"Could not write jpg file to {str(jpg_path)}")
    return jpg_path


def _convert_to_jpg(input_filename):
    """Convert a list of file or file to jpg"""
    if isinstance(input_filename, list):
        return list(map(_to_jpg, input_filename))
    return _to_jpg(input_filename)


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    images_dir = current_dir / "data" / "images"
    list_files = [f for f in images_dir.iterdir() if f.suffix == ".tif"]
    new_files = _convert_to_jpg(list_files)

    for ind, elem in enumerate(list_files):
        if elem.stem == new_files[ind].stem:
            os.remove(elem)
