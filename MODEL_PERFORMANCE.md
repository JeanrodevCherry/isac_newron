# Model performance index

Checkpoint comparison performed on 2026-11-24. No checkpoint files were removed or overwritten.

## How to interpret the comparison

- Pattern IoU and border F-score are measured on the same 26 images in `data/noise_filter/images` against `data/noise_filter/masks_refined`. These include training images for some checkpoints and are **not a strict held-out leaderboard**.
- Segmentation IoU and Dice are per-image means on the same 36 labeled JPG/LabelMe JSON pairs in `data/segmentation/images`. Each candidate segmentation checkpoint used `pattern_model_honeycomb_refined_v2.pt` for its pattern crop and mask channel, with threshold 0.5. Crop retention, however, was only 96.99% averaged across this corpus; inspect per-image crop loss before treating the means as fully comparable.
- Checkpoint train losses below are training losses, not validation losses. Losses from different loss functions/task configurations are not directly comparable. “Accuracy” is not reported: pixel accuracy is misleading for sparse foreground segmentation and is not implemented in the project evaluation. Prefer IoU/Dice, border F-score, crop retention, missed detections, component count, and visual review.
- The five external folders have no matching JSON annotations, so there is no measured external IoU/Dice. External summaries below are inference-only: detection rate, average mask coverage, predicted foreground fraction and connected components are diagnostics, not accuracy.

## Pattern checkpoints

| Checkpoint | Input | Best / final train loss | IoU | Border F | Pattern labels |
|---|---:|---:|---:|---:|---:|
| `pattern_model_honeycomb_refined_v2.pt` | 384 | 0.0625 / 0.0688 | **0.98046** | **0.99050** | 26 |
| `pattern_model_768.pt` | 768 | 0.0597 / 0.0622 | 0.98044 | 0.97283 | 26 |
| `pattern_model.pt` | 384 | 0.0771 / 0.0829 | 0.97959 | 0.99051 | 26 |
| `pattern_model_refined_exposure_150.pt` | 768 | 0.0674 / 0.0709 | 0.97827 | 0.96701 | 26 |
| `pattern_model_honeycomb_refined.pt` | 384 | 0.1312 / 0.1351 | 0.97316 | 0.98744 | 26 |
| `pattern_model_overexposed_combined.pt` | 768 | 0.1267 / 0.1289 | 0.96938 | 0.93576 | 26 |
| `pattern_model_honeycomb_candidate.pt` | 768 | 0.1067 / 0.1101 | 0.96332 | 0.88550 | 26 |
Two additional names found in an earlier checkpoint inventory (`pattern_model_hummongus_epoch.pt`, `pattern_model_large_batch.pt`) were not present in the workspace when this comparison was run, so they are not ranked here.

**Provisional pattern choice:** `pattern_model_honeycomb_refined_v2.pt`. It ties for highest IoU and has the highest border F-score among the evaluated candidates. `pattern_model.pt` is essentially tied on border F and only 0.00087 IoU lower; the refined-v2 choice is therefore provisional until truly independent, labeled images are available.

## Segmentation checkpoints

| Checkpoint | Best / final train loss | IoU | Dice | Mean predicted-minus-GT components | Labeled images |
|---|---:|---:|---:|---:|---:|
| `segmentation_model_refined_150.pt` | 0.1445 / 0.1466 | 0.75489 | **0.80967** | -4.58 | 36 |
| `segmentation_model.pt` | 0.1270 / 0.1303 | 0.75069 | 0.79509 | -2.47 | 36 |
| `segmentation_model_honeycomb_refined_v2.pt` | 0.2318 / 0.2318 | 0.74373 | 0.78767 | -4.42 | 36 |
| `segmentation_model_refined.pt` | 0.4333 / 0.4333 | 0.35319 | 0.48549 | +41.06 | 36 |

**Provisional segmentation choice:** `segmentation_model_refined_150.pt` for highest mean IoU/Dice in this comparison. It improves IoU by 0.0042 and Dice by 0.0146 over `segmentation_model.pt`; the margin is modest, and component-count underprediction averages 4.58 components per image. Keep both until independent labeled QC confirms the choice. `segmentation_model_refined.pt` performed substantially worse and produced many excess components.

## External inference-only checks

Pattern masks were thresholded and reduced to their largest connected component by `pattern_region`. All folders lack JSON ground truth; detection rate and coverage must not be interpreted as accuracy.

| Folder | Images | `pattern_model.pt` found / coverage | `pattern_model_768.pt` found / coverage | `pattern_model_honeycomb_candidate.pt` found / coverage | `pattern_model_honeycomb_refined.pt` found / coverage | `pattern_model_honeycomb_refined_v2.pt` found / coverage | `pattern_model_overexposed_combined.pt` found / coverage | `pattern_model_refined_exposure_150.pt` found / coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `NOVO_11_02_plate_1/input_QC` | 24 | 24 / 0.059 | 24 / 0.058 | 24 / 0.063 | 24 / 0.058 | 24 / 0.058 | 24 / 0.061 | 24 / 0.057 |
| `NOVO_11_02_plate_2/input_QC` | 24 | 24 / 0.059 | 24 / 0.059 | 24 / 0.063 | 24 / 0.057 | 24 / 0.058 | 24 / 0.061 | 24 / 0.057 |
| `QIMR/D0/merged` | 24 | 22 / 0.020 | 24 / 0.123 | 24 / 0.001 | 24 / 0.051 | 24 / 0.058 | 24 / 0.189 | 24 / 0.188 |
| `QIMR/D3/merged` | 24 | 24 / 0.046 | 24 / 0.130 | 24 / 0.002 | 24 / 0.083 | 24 / 0.081 | 24 / 0.181 | 24 / 0.178 |
| `lebeling_test/testimages` | 19 | 19 / 0.065 | 19 / 0.066 | 19 / 0.067 | 19 / 0.066 | 19 / 0.066 | 19 / 0.068 | 19 / 0.065 |

The baseline is `pattern_model.pt`. Coverage is mean foreground fraction among detected masks. The differing QIMR coverage, especially the candidate's near-zero coverage and overexposed/exposure-150 models' much larger coverage, requires visual QC; none of these diagnostics establishes which outline matches the true pattern. Segmentation component counts (base / refined-150), from the fixed `pattern_model_honeycomb_refined_v2.pt` input and threshold 0.5, were respectively 57.8 / 56.7 for NOVO plate 1, 58.6 / 57.5 for NOVO plate 2, 54.2 / 23.3 for QIMR D0, 32.2 / 26.7 for QIMR D3, and 58.6 / 57.6 for labeling test. Component count depends on image scale and postprocessing; external folders have no annotations, so these are diagnostics only.

## Selection and next steps

1. Do not delete any checkpoint yet.
2. Treat the choices above as provisional: create a fixed, independent labeled validation set representative of NOVO and QIMR acquisition conditions.
3. For pattern selection, compare IoU and border F-score plus missed-pattern rate and coverage consistency on that holdout.
4. For segmentation selection, compare per-image IoU/Dice, false-positive/false-negative area, object count against reviewed annotations, and crop retention. Report distributions or confidence intervals, not only means.
5. Keep the same pattern checkpoint, preprocessing and threshold for all segmentation candidates. Re-run comparisons after any mask-fill or threshold/postprocessing changes.
6. Only after visual review and repeatable held-out wins should one checkpoint per stage be designated for production. Preserve the other checkpoints as rollback candidates until that point.
