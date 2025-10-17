import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2, glob, os
import numpy as np
from tqdm import tqdm
import tifffile

from src.porosity import CompactnessLoss

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
        return torch.sigmoid(self.out(d1))
# ----------------------

class BCEDiceCompactLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6, compact_weight=0.2):
        super().__init__()
        self.bce = torch.nn.BCELoss()
        self.compact = CompactnessLoss(weight=compact_weight)
        self.smooth = smooth

    def forward(self, preds, targets):
        bce = self.bce(preds, targets)
        intersection = (preds * targets).sum()
        dice = 1 - (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        comp = self.compact(preds)
        return bce + dice + comp

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
    def __init__(self, img_dir, mask_dir, size=256):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
        self.size = size

        # Map by basename (without "_label" suffix)
        img_keys = {os.path.splitext(os.path.basename(p))[0]: p for p in self.img_paths}
        mask_keys = {os.path.splitext(os.path.basename(p))[0].replace("_label", ""): p for p in self.mask_paths}
        self.pairs = [(img_keys[k], mask_keys[k]) for k in img_keys if k in mask_keys]

    def __len__(self):
        return len(self.pairs)

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
        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size))

        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return img, mask

# -----------------------
# 3. Training loop
# -----------------------
def train_model(img_dir, mask_dir, epochs=25, lr=1e-3, batch_size=2, model_path="pattern_model.pt"):
    dataset = PatternDataset(img_dir, mask_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SmallUNet()
    # loss_fn = nn.BCELoss()
    loss_fn = BCEDiceCompactLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total = 0
        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
        print(f"Loss: {total / len(loader.dataset):.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    return model


# -----------------------
# 4. Inference helper
# -----------------------
def predict_and_crop(model, image_path, size=256, threshold=0.5):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    inp = cv2.resize(img, (size, size))
    inp_t = torch.from_numpy(inp).float().unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        mask = model(inp_t)[0,0].numpy()
    mask = cv2.resize(mask, (w, h))
    mask_bin = (mask > threshold).astype(np.uint8)

    # Find bounding box of mask
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        print("⚠️ No pattern detected.")
        return img
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    cropped = img[y1:y2, x1:x2]
    return cropped

def show_prediction(model,image_path,size=256,threshold=0.5):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    inp = cv2.resize(img, (size, size))
    inp_t = torch.from_numpy(inp).float().unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        mask = model(inp_t)[0,0].numpy()
    mask = cv2.resize(mask, (w, h))
    mask_bin = (mask > threshold).astype(np.uint8)
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        print("⚠️ No pattern detected.")
        return img
    return mask_bin