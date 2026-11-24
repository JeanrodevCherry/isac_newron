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

### Performance index and provisional model selection

The measured comparison currently favors `pattern_model_honeycomb_refined_v2.pt` (pattern IoU 0.98046, border F-score 0.99050) and `segmentation_model_refined_150.pt` (segmentation IoU 0.75489, Dice 0.80967). These are provisional: pattern scores use the available 26-image dataset, not an independent holdout, and supplied external QC folders do not have ground-truth annotations for measured accuracy. Training loss is not validation loss, and pixel accuracy is not a useful selection metric for sparse masks. See [MODEL_PERFORMANCE.md](MODEL_PERFORMANCE.md) for all checkpoint results, external inference diagnostics, limitations, and a validation-based selection checklist. No models have been removed.

### Latest external sample comparison

A seeded random sample of 10 images (seed `20261124`) from the five supplied QC folders was run through the 2 leading pattern checkpoints crossed with the 2 leading segmentation checkpoints. The panels are under `debug_out/model_comparison/<pattern>__<segmentation>/`; the sample paths are recorded in `debug_out/model_comparison/sample.txt`. Each of the four combinations has 10 diagnostic panels and a `results.xlsx`. No sampled image had a matching JSON annotation, so IoU/Dice are unavailable and this run is a visual comparison only.

The sampled QIMR D3 images show the clearest recurring weakness, especially `D03_Plate_1_C5_ch00_SV.jpg` and `D03_Plate_1_C3_ch00_SV.jpg`. The pattern detector’s thresholded mask/crop is irregular or incomplete on these images; the segmentation model then receives a truncated/fragmented pattern mask as its second channel and a denoised crop that suppresses parts of the image. The source images also have strong background illumination/reflections and the detected fields differ in shape/position from the more regular NOVO and labeling-test examples. This points more toward domain/appearance and pattern-mask generalization than a segmentation-checkpoint-only failure. It is a likely explanation based on visual outputs, not a confirmed root cause: the sample has no ground truth, and the different pattern candidates do not fail identically. Inspect the cited panels and create reviewed pattern masks for representative QIMR images before changing the model or postprocessing.

Example panels for the repeatedly difficult samples:

- [`D03_Plate_1_C5_ch00_SV.jpg` — refined-v2 pattern + refined-150 segmentation](debug_out/model_comparison/pattern_model_honeycomb_refined_v2__segmentation_model_refined_150/D03_Plate_1_C5_ch00_SV.jpg)
- [`D03_Plate_1_C3_ch00_SV.jpg` — refined-v2 pattern + refined-150 segmentation](debug_out/model_comparison/pattern_model_honeycomb_refined_v2__segmentation_model_refined_150/D03_Plate_1_C3_ch00_SV.jpg)

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
# Pixel size per image = --pattern-um (6000) / detected pattern width in px (exports have different resolutions);
# --um-per-px overrides it with a fixed value. A warning is printed when the pattern is narrower than --min-pattern-mm (1.0)
```

`test_pipeline.py` checks (1) pattern IoU/border F-score, (2) the pattern crop contains all labelled spheroids,
(3) spheroid IoU/Dice, saves panels in `debug_out/test/` (green = ground truth, red = prediction) and exits with 1
if a mean IoU is under its threshold (`--pattern-iou 0.9`, `--spheroid-iou 0.6`).
On 6 held-out images the spheroid model gets mean IoU 0.89 / Dice 0.94.


Example of pipeline testing:
1. Simple ouput in debug_out
2. External dir and show segmentation && min area spheroids
```bash
python validate_pipeline.py --images /home/jpeiteado/Bureau/pripri_tmp/lebeling_test/testimages/
python validate_pipeline.py --images ~/Bureau/pripri_tmp/NOVO_11_02_plate_1/input_QC/ --show-segmented --out /home/jpeiteado/Bureau/pripri_tmp/temp/ --min-area-spheroid-px 600
```