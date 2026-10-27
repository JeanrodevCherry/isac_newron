import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2, glob, os
import numpy as np
from tqdm import tqdm
import tifffile
from datetime import datetime, timezone
from pathlib import Path
import json
import pdb

from src.porosity import BCETverskyBoundaryNoHoleLoss
from src.toolkit import pattern_region


class SpheroidDataset(Dataset):
    def __init__(self, img_dir, json_dir, honeycomb_mask_dir=None, size=512, augment=False,
                 pattern_model=None):
        # With pattern_model, samples are built exactly like at inference: the
        # image is denoised with the pattern mask, cropped around the pattern,
        # and the pattern mask is the 2nd channel (honeycomb_mask_dir is ignored).
        self.size = size
        self.pattern_model = pattern_model
        self.augment = augment
        self.honeycomb_mask_dir = honeycomb_mask_dir

        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        json_keys = {
            os.path.splitext(os.path.basename(p))[0].replace("_label", ""): p
            for p in glob.glob(os.path.join(json_dir, "*.json"))
        }
        self.pairs = [
            (p, json_keys[os.path.splitext(os.path.basename(p))[0]])
            for p in self.img_paths
            if os.path.splitext(os.path.basename(p))[0] in json_keys
        ]
        self.cache = [self._prepare(*pair) for pair in self.pairs] if pattern_model else None

    def _prepare(self, img_path, json_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        sph = self._load_spheroid_mask(json_path, img.shape)
        pmask, box, _ = pattern_region(self.pattern_model, img)
        if pmask is None:
            raise RuntimeError(f"Pattern model found no pattern in {img_path}")
        x1, y1, x2, y2 = box
        crop = (img * pmask)[y1:y2, x1:x2]
        interp = dict(interpolation=cv2.INTER_NEAREST)
        sz = (self.size, self.size)
        return (cv2.resize(crop, sz, interpolation=cv2.INTER_AREA),
                cv2.resize(sph[y1:y2, x1:x2], sz, **interp).astype(np.float32),
                cv2.resize(pmask[y1:y2, x1:x2], sz, **interp).astype(np.float32))

    def _load_spheroid_mask(self, json_path: str, shape_hw: tuple[int, int]) -> np.ndarray:
        """Rasterize labelme polygon annotations into a binary mask."""
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)

        with open(json_path) as f:
            data = json.load(f)

        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "polygon":
                print(f"Warning: skipping shape_type {shape.get('shape_type')!r} in {json_path}")
                continue
            pts = np.array(shape["points"], dtype=np.float32)
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)

        return mask

    def _augment_pair(self, img, masks):  # masks: list of arrays augmented together
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
            masks = [cv2.flip(m, 1) for m in masks]
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 0)
            masks = [cv2.flip(m, 0) for m in masks]
        if np.random.rand() < 0.5:
            k = np.random.randint(0, 4)
            img = np.rot90(img, k).copy()
            masks = [np.rot90(m, k).copy() for m in masks]
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-35.0, 35.0)
            center = (self.size / 2.0, self.size / 2.0)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, matrix, (self.size, self.size),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            masks = [cv2.warpAffine(m, matrix, (self.size, self.size),
                                    flags=cv2.INTER_NEAREST, borderValue=0) for m in masks]
        if np.random.rand() < 0.8:
            alpha = np.random.uniform(0.8, 1.25)
            beta = np.random.uniform(-22.0, 22.0)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255)
        if np.random.rand() < 0.4:
            gamma = np.random.uniform(0.75, 1.35)
            img = np.clip(((img / 255.0) ** gamma) * 255.0, 0, 255)
        return img.astype(np.uint8), masks

    def __getitem__(self, i):
        if self.cache is not None:
            img, spheroid_mask, hc = self.cache[i]
            if self.augment:
                img, [spheroid_mask, hc] = self._augment_pair(img, [spheroid_mask, hc])
            x = torch.cat([torch.from_numpy(img).float().unsqueeze(0) / 255.0,
                           torch.from_numpy(hc).float().unsqueeze(0)], dim=0)
            return x, torch.from_numpy(spheroid_mask).float().unsqueeze(0)

        img_path, json_path = self.pairs[i]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape

        spheroid_mask = self._load_spheroid_mask(json_path, (h, w))

        # Honeycomb mask: ground truth if available, otherwise ones (no gating)
        # pdb.set_trace()
        if self.honeycomb_mask_dir:
            hc_name = os.path.splitext(os.path.basename(img_path))[0] +".tif"#+ "_label.tif"
            hc = tifffile.imread(os.path.join(self.honeycomb_mask_dir, hc_name))
            if hc.ndim > 2:
                hc = hc[..., 0]
            hc = (hc > 0).astype(np.float32)
        else:
            hc = np.ones((h, w), dtype=np.float32)

        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        spheroid_mask = cv2.resize(spheroid_mask, (self.size, self.size),
                                   interpolation=cv2.INTER_NEAREST).astype(np.float32)
        hc = cv2.resize(hc, (self.size, self.size),
                        interpolation=cv2.INTER_NEAREST).astype(np.float32)

        if self.augment:
            img, [spheroid_mask, hc] = self._augment_pair(img, [spheroid_mask, hc])

        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0        # (1, H, W)
        hc = torch.from_numpy(hc).float().unsqueeze(0)                  # (1, H, W)
        target = torch.from_numpy(spheroid_mask).float().unsqueeze(0)   # (1, H, W)
        x = torch.cat([img, hc], dim=0)                                 # (2, H, W)
        return x, target
    def __len__(self):
        return len(self.pairs)





class PatternDataset(Dataset):
    """Pattern (honeycomb region) dataset with strong augmentation.

    Images are resized once (INTER_AREA) and cached; every __getitem__ applies
    random flips/rotations/affine (scale, shift), illumination changes, blur,
    sensor noise and random dark/bright patches (simulated debris).
    """

    def __init__(self, img_dir, mask_dir, size=512, augment=False, names=None):
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif*")))
        self.size = size
        self.augment = augment

        img_keys = {os.path.splitext(os.path.basename(p))[0].replace(" ", "_"): p for p in img_paths}
        mask_keys = {os.path.splitext(os.path.basename(p))[0].replace("_label", "").replace(" ", "_"): p
                     for p in mask_paths}
        self.names = sorted(k for k in img_keys if k in mask_keys and (names is None or k in names))
        self.pairs = [(img_keys[k], mask_keys[k]) for k in self.names]
        self.cache = [self._load(*pair) for pair in self.pairs]

    def __len__(self):
        return len(self.pairs)

    def subset(self, names, augment):
        """Cheap view on already-cached samples."""
        import copy
        sub = copy.copy(self)
        idx = [i for i, n in enumerate(self.names) if n in names]
        sub.names = [self.names[i] for i in idx]
        sub.pairs = [self.pairs[i] for i in idx]
        sub.cache = [self.cache[i] for i in idx]
        sub.augment = augment
        return sub

    def _load(self, img_path, mask_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = tifffile.imread(mask_path)
        if mask.ndim > 2:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return img, mask

    def _augment_pair(self, img, mask):
        S = self.size
        rng = np.random
        if rng.rand() < 0.5:
            img, mask = cv2.flip(img, 1), cv2.flip(mask, 1)
        if rng.rand() < 0.5:
            img, mask = cv2.flip(img, 0), cv2.flip(mask, 0)
        k = rng.randint(0, 4)
        img, mask = np.rot90(img, k).copy(), np.rot90(mask, k).copy()

        if rng.rand() < 0.8:  # rotation + scale + translation
            m = cv2.getRotationMatrix2D((S / 2.0, S / 2.0), rng.uniform(-35, 35), rng.uniform(0.8, 1.25))
            m[:, 2] += rng.uniform(-0.1, 0.1, 2) * S
            img = cv2.warpAffine(img, m, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, m, (S, S), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)

        img = img.astype(np.float32)
        if rng.rand() < 0.8:
            img = img * rng.uniform(0.7, 1.4) + rng.uniform(-25, 25)
        if rng.rand() < 0.3:  # vignetting / uneven illumination
            yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S - 0.5
            gx, gy = rng.uniform(-0.6, 0.6, 2)
            img = img * (1.0 + gx * xx + gy * yy)
        if rng.rand() < 0.4:
            img = 255.0 * (np.clip(img, 0, 255) / 255.0) ** rng.uniform(0.7, 1.4)
        if rng.rand() < 0.3:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 2.0))
        if rng.rand() < 0.5:
            img = img + rng.normal(0, rng.uniform(1, 8), img.shape)
        for _ in range(rng.randint(0, 4) if rng.rand() < 0.3 else 0):  # debris-like patches
            r = rng.randint(S // 40, S // 10)
            cx, cy = rng.randint(0, S, 2)
            patch = np.zeros((S, S), np.uint8)
            cv2.circle(patch, (cx, cy), r, 1, -1)
            patch = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), r / 3.0)
            img = img * (1 - patch) + rng.uniform(0, 255) * patch
        return np.clip(img, 0, 255).astype(np.uint8), mask

    def __getitem__(self, i):
        img, mask = self.cache[i]
        if self.augment:
            img, mask = self._augment_pair(img, mask)
        mask = (mask > 0.5).astype(np.float32)
        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return img, torch.from_numpy(mask).unsqueeze(0)
