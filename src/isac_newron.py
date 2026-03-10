import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2, glob, os
import numpy as np
from tqdm import tqdm
import tifffile
from datetime import datetime, timezone

from src.porosity import BCETverskyBoundaryNoHoleLoss


def extract_state_dict(loaded_checkpoint):
    """
    Backward-compatible extractor:
    - old format: plain state_dict
    - new format: checkpoint dict with model_state_dict
    """
    if isinstance(loaded_checkpoint, dict) and "model_state_dict" in loaded_checkpoint:
        return loaded_checkpoint["model_state_dict"]
    return loaded_checkpoint

# -----------------------
# 1. Tiny U-Net backbone
# -----------------------
class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_bn(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        self.enc1 = conv_bn(1, 16)
        self.enc2 = conv_bn(16, 32)
        self.enc3 = conv_bn(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.dec2 = conv_bn(64 + 32, 32)
        self.dec1 = conv_bn(32 + 16, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = torch.cat([nn.functional.interpolate(e3, scale_factor=2, mode="bilinear"), e2], 1)
        d2 = self.dec2(d2)
        d1 = torch.cat([nn.functional.interpolate(d2, scale_factor=2, mode="bilinear"), e1], 1)
        d1 = self.dec1(d1)
        return self.out(d1)
# ----------------------

class BoundaryAwareSegLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = BCETverskyBoundaryNoHoleLoss(
            tversky_weight=1.0,
            boundary_weight=1.2,
            no_hole_weight=1.8,
        )

    def forward(self, preds, targets):
        return self.loss(preds, targets)

# -----------------------
# 2. Dataset loader
# -----------------------
# class PatternDataset(Dataset):
#     def __init__(self, img_dir, mask_dir, size=256):
#         self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
#         self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
#         self.size = size
#         print(self.img_paths)
#         print(self.mask_paths)

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, i):
#         img = cv2.imread(self.img_paths[i], cv2.IMREAD_GRAYSCALE)
#         mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_UNCHANGED)
#         if mask.ndim == 3:
#             mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#         mask = mask.astype(np.float32)
#         if mask.max() > 1.0:   # normalize if 0–255 or 0–65535
#             mask /= mask.max()
#         img = cv2.resize(img, (self.size, self.size))
#         mask = cv2.resize(mask, (self.size, self.size))
#         img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
#         mask = torch.from_numpy(mask).float().unsqueeze(0)
#         return img, mask
class PatternDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=256, augment=False):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
        self.size = size
        self.augment = augment

        # Map by basename (without "_label" suffix)
        img_keys = {os.path.splitext(os.path.basename(p))[0]: p for p in self.img_paths}
        mask_keys = {os.path.splitext(os.path.basename(p))[0].replace("_label", ""): p for p in self.mask_paths}
        self.pairs = [(img_keys[k], mask_keys[k]) for k in img_keys if k in mask_keys]

    def __len__(self):
        return len(self.pairs)

    def _augment_pair(self, img, mask):
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)

        if np.random.rand() < 0.5:
            k = np.random.randint(0, 4)
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()

        if np.random.rand() < 0.5:
            angle = np.random.uniform(-35.0, 35.0)
            center = (self.size / 2.0, self.size / 2.0)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(
                img,
                matrix,
                (self.size, self.size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            mask = cv2.warpAffine(
                mask,
                matrix,
                (self.size, self.size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        if np.random.rand() < 0.8:
            alpha = np.random.uniform(0.8, 1.25)
            beta = np.random.uniform(-22.0, 22.0)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255)

        if np.random.rand() < 0.4:
            gamma = np.random.uniform(0.75, 1.35)
            img = np.clip(((img / 255.0) ** gamma) * 255.0, 0, 255)

        return img.astype(np.uint8), mask.astype(np.float32)

    def __getitem__(self, i):
        img_path, mask_path = self.pairs[i]

        # --- Read JPG ---
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # --- Read TIFF mask (float or int) ---
        mask = tifffile.imread(mask_path)
        if mask.ndim > 2:
            mask = mask[..., 0]  # drop extra channels if any
        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask /= mask.max()

        # --- Resize & normalize ---
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, mask = self._augment_pair(img, mask)

        mask = (mask > 0.5).astype(np.float32)

        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return img, mask

# -----------------------
# 3. Training loop
# -----------------------
def train_model(
    img_dir,
    mask_dir,
    epochs=25,
    lr=1e-2,
    batch_size=2,
    model_path="pattern_model.pt",
    augment=True,
    device=None,
):
    """
    train_model train model using input data and custom LossFunction

    Parameters
    ----------
    img_dir : Path
        training input dataset folder
    mask_dir : Path
        training verification dataset folder
    epochs : int, optional
        epochs to perform, by default 25
    lr : _type_, optional
        optimizer lr param, by default 1e-2
    batch_size : int, optional
        maximum batch size, by default 2
    model_path : str, optional
        model checkpoint path, by default "pattern_model.pt"
    augment : bool, optional
        apply illumination/orientation augmentations, by default True
    device : str or torch.device, optional
        training device. If None, uses CUDA when available, otherwise CPU

    Returns
    -------
    model
        SmallUnet type model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    dataset = PatternDataset(img_dir, mask_dir, augment=augment)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
    )

    model = SmallUNet().to(device)
    loss_fn = BoundaryAwareSegLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda) if use_cuda else None

    lr_drop_epoch = 50
    lr_after_drop = 1e-3

    print(f"🚀 Training device: {device}")
    epoch_losses = []

    for epoch in range(epochs):
        if epoch == lr_drop_epoch:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_after_drop
            print(f"🔁 Learning rate changed to {lr_after_drop} at epoch {epoch+1}")

        model.train()
        total = 0
        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)

            optimizer.zero_grad()

            if use_cuda:
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    pred = model(x)
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
            "loss": "BCETverskyBoundaryNoHoleLoss",
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


# -----------------------
# 4. Inference helper
# -----------------------
# def predict_and_crop(model, image_path, size=256, threshold=0.5):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     h, w = img.shape
#     inp = cv2.resize(img, (size, size))
#     inp_t = torch.from_numpy(inp).float().unsqueeze(0).unsqueeze(0) / 255.0
#     with torch.no_grad():
#         mask = model(inp_t)[0,0].numpy()
#     mask = cv2.resize(mask, (w, h))
#     mask_bin = (mask > threshold).astype(np.uint8)

#     # Find bounding box of mask
#     ys, xs = np.where(mask_bin > 0)
#     if len(xs) == 0:
#         print("⚠️ No pattern detected.")
#         return img
#     x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
#     cropped = img[y1:y2, x1:x2]
#     return cropped
def predict_and_crop(model, image_path, size=256, threshold=0.5, expand_ratio=1.05):
    """
    Predict mask and crop around the *largest connected region*.
    - Keeps only the largest component.
    - Centers a square crop around it.
    - expand_ratio enlarges crop slightly beyond the detected region.
    """
    # --- Load image ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape

    # --- Model inference ---
    inp = cv2.resize(img, (size, size))
    device = next(model.parameters()).device
    inp_t = (torch.from_numpy(inp).float().unsqueeze(0).unsqueeze(0) / 255.0).to(device)
    with torch.no_grad():
        logits = model(inp_t)
        mask = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    mask = cv2.resize(mask, (w, h))
    mask_bin = (mask > threshold).astype(np.uint8)

    # --- Find connected components ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num_labels <= 1:
        print("⚠️ No pattern detected.")
        return img

    # --- Select largest region (ignore background index 0) ---
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, bw, bh, area = stats[largest_idx]
    cx, cy = centroids[largest_idx]

    # --- Compute square crop around the region ---
    r = int(max(bw, bh) * expand_ratio / 2)
    cx, cy = int(cx), int(cy)
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r)
    y2 = min(h, cy + r)
    cropped = img[y1:y2, x1:x2]
    return cropped
def parse_image(model, image_path, size=256, threshold=0.5, expand_ratio=1.05):
    """
    Predict mask and crop around the *largest connected region*.
    - Keeps only the largest component.
    - Centers a square crop around it.
    - expand_ratio enlarges crop slightly beyond the detected region.
    - Mask should be same size than the cropped image, to fit correctly.

    returns
    -------

    cropped
        cropped jpg image (numpy.ndarray) with size (w,h)
    mask
        inverted mask with size (w,h)
    """
    # --- Load image ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape

    # --- Model inference ---
    inp = cv2.resize(img, (size, size))
    device = next(model.parameters()).device
    inp_t = (torch.from_numpy(inp).float().unsqueeze(0).unsqueeze(0) / 255.0).to(device)
    with torch.no_grad():
        logits = model(inp_t)
        mask = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    mask = cv2.resize(mask, (w, h))
    mask_bin = (mask > threshold).astype(np.uint8)

    # --- Find connected components ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num_labels <= 1:
        print("⚠️ No pattern detected.")
        return img

    # --- Select largest region (ignore background index 0) ---
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, bw, bh, area = stats[largest_idx]
    cx, cy = centroids[largest_idx]

    # --- Compute square crop around the region ---
    r = int(max(bw, bh) * expand_ratio / 2)
    cx, cy = int(cx), int(cy)
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r)
    y2 = min(h, cy + r)
    cropped = img[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    # return cropped,invert_mask(cropped_mask)
    return cropped, cropped_mask
def show_prediction(model,image_path,size=256,threshold=0.5):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    inp = cv2.resize(img, (size, size))
    device = next(model.parameters()).device
    inp_t = (torch.from_numpy(inp).float().unsqueeze(0).unsqueeze(0) / 255.0).to(device)
    with torch.no_grad():
        logits = model(inp_t)
        mask = torch.sigmoid(logits)[0,0].detach().cpu().numpy()
    mask = cv2.resize(mask, (w, h))
    mask_bin = (mask > threshold).astype(np.uint8)
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        print("⚠️ No pattern detected.")
        return img
    return invert_mask(mask_bin)

def invert_mask(mask):
    """
    invert_mask not binary, invert the intensity

    Parameters
    ----------
    mask : numpy.array
        mask loaded from tif file or output of NNN

    Returns
    -------
    inverted mask
        Binary -> Invert over 1 - 0
        0 - 255 -> Invert intensity
    """
    if mask.max() >1:
        return (255 - mask).astype(mask.dtype)
    return (1 - mask).astype(mask.dtype)