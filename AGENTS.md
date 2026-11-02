# Agent Guidelines

## Project purpose
This project segments spheroids in microscope images using a two-stage U-Net pipeline: pattern detection followed by spheroid segmentation. Preserve this pipeline unless the requested change explicitly alters it.

## Before modifying
- Inspect the relevant source, tests, and callers before editing.
- Check whether changes affect inference, training, data preparation, or model checkpoint compatibility.
- Keep generated data, debug output, and model checkpoints out of source changes unless explicitly requested.

## Data and models
- `data/` is intended to be git-ignored; verify repository ignore configuration before relying on that assumption.
- Model checkpoints may be tracked through Git LFS and include model state/configuration; inspect current loading code and avoid breaking existing checkpoints.
- Preserve the expected image sizes, channels, and preprocessing shared by training and inference.
- Training and validation require local datasets and model files; do not assume they are available in every environment.

## Validation
- Run targeted tests, then `python test_pipeline.py` when applicable and required data and models are available.
- For visual pipeline changes, run `python validate_pipeline.py` if the required data and models are available.
- Report checks that could not run and identify missing data or models.

## Change conventions
- Follow existing Python style and reuse project dependencies.
- Keep changes focused; do not modify labels, datasets, generated output, or checkpoints as a side effect.
- Update usage or training instructions in `README.md` when behavior or commands change.


## Environment
- The application is running on python3, the venv is located in ./venv/
- ASK before installing new deps
- Follow python pep8 code standard, BLACK formatting and mypy validation when possible, do not alter existing code with formatting modification, only new code
