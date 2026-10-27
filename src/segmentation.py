import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2, glob, os
import numpy as np
from tqdm import tqdm
import tifffile
from pathlib import Path
from datetime import datetime, timezone

from src.models import SpheroidUNet, load_pattern_model, load_spheroid_model
from src.toolkit import build_overlay, pattern_region
from src.porosity import DiceBCELoss
from src.datasets import SpheroidDataset

SPHEROID_SIZE = 512


def train_segmentation_model(
    img_dir,
    json_dir,
    mask_dir=None,          # honeycomb masks (TIF) for channel 2; optional
    epochs=25,
    lr=1e-3,
    batch_size=2,
    model_path: str | Path = "segmentation_model.pt",
    augment=True,
    device=None,
    pattern_model_path="pattern_model.pt",
):
    """
    Train the spheroid segmentation model (SpheroidUNet) with Dice+BCE loss.

    Parameters
    ----------
    img_dir : Path
        Folder with training images (JPG).
    json_dir : Path
        Folder with labelme JSON annotations (spheroid polygons).
    mask_dir : Path, optional
        Folder with honeycomb masks (TIF) used as second input channel.
    epochs : int, optional
    lr : float, optional
    batch_size : int, optional
    model_path : str or Path, optional
        Checkpoint path, by default "segmentation_model.pt".
    augment : bool, optional
        Apply orientation/illumination augmentations, by default True.
    device : str or torch.device, optional
        If None, uses CUDA when available.

    Returns
    -------
    SpheroidUNet
        Trained model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    pattern_model, _, _ = load_pattern_model(pattern_model_path)
    pattern_model.to(device)
    with torch.no_grad():
        dataset = SpheroidDataset(
            img_dir, json_dir, honeycomb_mask_dir=mask_dir, size=SPHEROID_SIZE,
            augment=augment, pattern_model=pattern_model,
        )
    if len(dataset) == 0:
        raise RuntimeError(
            "No image/JSON pairs found — check img_dir/json_dir contents and naming."
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
    )

    model = SpheroidUNet().to(device)
    # model = SmallUNet().to(device)
    loss_fn = DiceBCELoss(
        dice_weight=1.0,
        bce_weight=1.0,
        pos_weight=torch.tensor([5.0]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda) if use_cuda else None

    lr_drop_epoch = 50
    lr_after_drop = 1e-4

    print(f"🚀 Training device: {device} | dataset size: {len(dataset)}")
    epoch_losses = []

    for epoch in range(epochs):
        if epoch == lr_drop_epoch:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_after_drop
            print(f"🔁 Learning rate changed to {lr_after_drop} at epoch {epoch+1}")

        model.train()
        total = 0.0
        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)

            optimizer.zero_grad()

            if use_cuda:
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    pred = model(x)          # logits (no sigmoid!)
                    loss = loss_fn(pred, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()

            total += loss.item() * x.size(0)

        epoch_loss = total / len(loader.dataset)
        epoch_losses.append(float(epoch_loss))
        print(f"Loss: {epoch_loss:.4f}")

    best_loss = min(epoch_losses)
    best_epoch = int(np.argmin(epoch_losses) + 1)
    last_k = min(5, len(epoch_losses))
    mean_last_k = float(np.mean(epoch_losses[-last_k:]))

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "training_quality": {
            "epoch_losses": epoch_losses,
            "final_train_loss": float(epoch_losses[-1]),
            "best_train_loss": float(best_loss),
            "best_epoch": best_epoch,
            "mean_last5_train_loss": mean_last_k,
            "epochs": int(epochs),
            "dataset_size": int(len(dataset)),
        },
        "train_config": {
            "lr": float(lr),
            "lr_drop_epoch": int(lr_drop_epoch),
            "lr_after_drop": float(lr_after_drop),
            "batch_size": int(batch_size),
            "augment": bool(augment),
            "device": str(device),
            "loss": "DiceBCELoss",          # was BCETverskyBoundaryNoHoleLoss (stale)
        },
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    torch.save(checkpoint, model_path)
    print(f"✅ Model saved to {model_path}")
    print(
        f"📊 Quality -> final: {epoch_losses[-1]:.4f}, "
        f"best: {best_loss:.4f} (epoch {best_epoch}), "
        f"mean_last5: {mean_last_k:.4f}"
    )
    return model


def predict_mask_full(model, image, size=256, threshold=0.5):
    """Binary mask of a 1-channel model at ORIGINAL resolution (no cropping)."""
    h, w = image.shape[:2]
    inp = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    inp_t = torch.from_numpy(inp).float().unsqueeze(0).unsqueeze(0) / 255.0
    inp_t = inp_t.to(next(model.parameters()).device)
    with torch.no_grad():
        mask = torch.sigmoid(model(inp_t))[0, 0].detach().cpu().numpy()
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return (mask > threshold).astype(np.uint8)


def predict_spheroid_prob(model, crop, pattern_crop, size=SPHEROID_SIZE):
    """Spheroid probability map (crop resolution) from a denoised crop + pattern gate."""
    ch, cw = crop.shape
    img = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    hc = cv2.resize(pattern_crop.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
    x = torch.from_numpy(np.stack([img.astype(np.float32) / 255.0, hc]))[None]
    x = x.to(next(model.parameters()).device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return cv2.resize(prob, (cw, ch), interpolation=cv2.INTER_LINEAR)


def apply_segmentation_model_to_image(model_path, image_path, size=SPHEROID_SIZE,
                                      threshold=0.5, expand_ratio=1.15,
                                      pattern_model_path="pattern_model.pt",
                                      device=None):
    """Pattern model (denoise + crop) -> spheroid model. Returns (denoised crop, probability map)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)

    pattern_model, _, _ = load_pattern_model(pattern_model_path)
    pattern_model.to(device)
    pmask, box, _ = pattern_region(pattern_model, img, threshold=threshold,
                                   expand_ratio=expand_ratio)
    if pmask is None:
        print("⚠️ Pattern model found no valid region — check image/illumination.")
        return img, np.zeros(img.shape, dtype=np.float32)

    x1, y1, x2, y2 = box
    denoised = (img * pmask)[y1:y2, x1:x2]
    model, _, _ = load_spheroid_model(model_path=model_path)
    model.to(device)
    prob = predict_spheroid_prob(model, denoised, pmask[y1:y2, x1:x2], size=size)
    return denoised, prob

def segment_image(model_path, image_path, output_path=None):
    crop, label = apply_segmentation_model_to_image(model_path, image_path)

    overlay, mask_vis = build_overlay(crop, label)

    if output_path is None:
        p = Path(image_path).resolve()
        output_path = p.with_stem(p.stem + "_segmented").with_suffix(".tif")
    cv2.imwrite(str(output_path), (label * 255).astype(np.uint8))
    return crop, label