# Isaac Newron _  U-NET

Internal use only, not even licensed yet

___


## Installation

```bash
git clone git@github.com:JeanrodevCherry/isac_newron.git
cd ./isac_newron/
git lfs fetch origin main
python -m venv venv
```


<details>
  <summary>Windows installation</summary>
    
```bash
source ./venv/Scripts/activate
pip install -r ./requirements.txt
```

</details>

<details>
  <summary>Linux installation</summary>

```bash
source ./venv/bin/activate
pip install -r ./requirements.txt
```
</details>

## What it does

Microscope images of micro-well plates contain a honeycomb pattern of wells, each holding one
spheroid, surrounded by noise (well rim, debris, illumination gradients). The project runs two U-Nets in sequence:

1. **Pattern model** (`pattern_model.pt`, `PatternUNet`, 384 px input) finds the honeycomb region.
   Only the largest connected region is kept; everything outside is zeroed (denoising) and the image is cropped around it.
2. **Spheroid model** (`segmentation_model.pt`, `SpheroidUNet`, 512 px input) takes the denoised crop plus the
   pattern mask as a second channel and predicts the spheroid mask.

```
image -> pattern model -> largest region -> denoise + crop -> spheroid model -> spheroid mask
```

## Usage

Activate virtual environment:

```bash
source ./venv/bin/activate # linux
source ./venv/Scripts/activate #windows
```

```bash
python ./__init__.py -filename "<image.jpg>"             # pattern only: cropped_result.jpg, overlay_result.jpg, mask
python ./__init__.py --segment -filename "<image.jpg>"    # full pipeline -> <image>_segmented.tif
python train_pipeline.py --model segmentation            # retrain the spheroid model
```

From Python: `from src.segmentation import segment_image` -> `crop, probability_map = segment_image("segmentation_model.pt", path)`.

### Model files

`pattern_model.pt` and `segmentation_model.pt` are checkpoints (`model_state_dict`, `training_quality`, `train_config`).
Make sure they are loaded with lfs. The pattern checkpoint stores its architecture and input size.

## Data layout (`data/` is git-ignored)

- `data/noise_filter/images`, `data/noise_filter/masks`: images with hand-drawn pattern masks (`<stem>_label.tif`).
- `data/noise_filter/masks_refined`: pattern masks used for training (1/8 resolution, lattice-based, accurate borders).
- `data/segmentation/images`: `<stem>.jpg` + labelme `<stem>.json` (spheroid polygons).

## Training

### Pattern model

```bash
python train_pipeline.py --model pattern --epochs 30 --size 384 --out pattern_model.pt
python train_pipeline.py --model pattern --val --out /tmp/p.pt  # hold out 5 images and compare with current model
```

`src/pattern.py` trains `PatternUNet` with a border-aware loss (BCE weighted around the mask edge + Dice + edge
consistency). `PatternDataset` augments every sample on the fly: flips, rotations, scale/translation, brightness/contrast,
vignetting, gamma, blur, noise and random debris-like patches. On 5 held-out images IoU went 0.87 -> 0.98 and
border F-score 0.85 -> 0.995 versus the previous `SmallUNet`.

Pattern training reads `data/noise_filter/images` and `data/noise_filter/masks_refined` by default. Override these with
`--pattern-img-dir` and `--pattern-mask-dir`.

### Labelling spheroids automatically

```bash
python label_noise_filter.py --write
```

`src/labeling.py` finds dark spheroids, keeps those on the hexagonal well lattice and writes labelme JSONs
(+ copies of the images) into `data/segmentation/images`, and the refined pattern masks into
`data/noise_filter/masks_refined`. Images where fewer than ~50 wells are found are skipped.
QA sheet: `debug_out/label_sheet.jpg`. The automatic labels reach IoU 0.81-0.91 against the manual ones, so review the sheet.
`Plate_3_*` images are already denoised; their spheroids come from the existing masks.

### Spheroid model

```bash
python train_pipeline.py --model segmentation --epochs 25 --out segmentation_model.pt
```

By default, segmentation training reads images and labelme JSONs from `data/segmentation/images` and uses
`pattern_model.pt` to prepare training samples exactly as at inference (pattern model, denoise, crop). Use
`--segmentation-img-dir`, `--segmentation-json-dir`, `--segmentation-mask-dir`, and `--pattern-model` to override
those paths. Both training modes accept `--epochs`, `--batch-size`, `--lr`, and `--out`; `--size` and `--val` apply
to pattern training.

## Testing

```bash
python test_pipeline.py                      # all images
python test_pipeline.py --holdout <stem> ... # evaluate only images not used in training
python validate_pipeline.py                  # quick 7-step visual panels
```

`test_pipeline.py` checks (1) pattern IoU/border F-score, (2) the pattern crop contains all labelled spheroids,
(3) spheroid IoU/Dice, saves panels in `debug_out/test/` (green = ground truth, red = prediction) and exits with 1
if a mean IoU is under its threshold (`--pattern-iou 0.9`, `--spheroid-iou 0.6`).
On 6 held-out images the spheroid model gets mean IoU 0.89 / Dice 0.94.
