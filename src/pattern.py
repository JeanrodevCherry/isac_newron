import numpy as np
import torch
from datetime import datetime, timezone
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import PatternDataset
from src.models import PatternUNet
from src.porosity import BorderAwareBCEDiceLoss


def evaluate_pattern_model(model, dataset, tol=3):
    """Mean IoU and border F-score (tolerance `tol` px at model resolution) on un-augmented data."""
    import cv2
    model.eval()
    ious, fs = [], []
    k = np.ones((3, 3), np.uint8)
    with torch.no_grad():
        for img, gt in dataset:
            pred = (torch.sigmoid(model(img[None]))[0, 0].numpy() > 0.5).astype(np.uint8)
            gt = gt[0].numpy().astype(np.uint8)
            ious.append((pred & gt).sum() / max((pred | gt).sum(), 1))
            eb = lambda m: m - cv2.erode(m, k)
            pb, gb = eb(pred), eb(gt)
            pd_, gd = cv2.dilate(pb, k, iterations=tol), cv2.dilate(gb, k, iterations=tol)
            prec = (pb & gd).sum() / max(pb.sum(), 1)
            rec = (gb & pd_).sum() / max(gb.sum(), 1)
            fs.append(2 * prec * rec / max(prec + rec, 1e-9))
    return float(np.mean(ious)), float(np.mean(fs))


def train_pattern_model(img_dir, mask_dir, epochs=150, lr=2e-3, batch_size=4, size=512,
                        model_path="pattern_model.pt", augment=True, device=None, val_names=None):
    """Train PatternUNet. If val_names is given those images are held out and reported."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_ds = PatternDataset(img_dir, mask_dir, size=size, augment=False)
    train_names = [n for n in all_ds.names if not val_names or n not in val_names]
    train_ds = all_ds.subset(train_names, augment)
    val_ds = all_ds.subset(val_names, False) if val_names else None
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size)

    model = PatternUNet().to(device)
    model.input_size = size
    loss_fn = BorderAwareBCEDiceLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # each epoch sees every image several times with different augmentations
    reps = max(1, 64 // max(len(train_ds), 1))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader) * reps)
    print(f"🚀 device={device} train={len(train_ds)} val={len(val_ds) if val_ds else 0} size={size}")
    losses = []
    bar = tqdm(range(epochs), desc="Pattern training", unit="epoch")
    for epoch in bar:
        model.train()
        total, n = 0.0, 0
        for _ in range(reps):
            for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item() * x.size(0)
                n += x.size(0)
        losses.append(total / n)
        msg = f"epoch {epoch+1}/{epochs} loss={losses[-1]:.4f}"
        if val_ds and ((epoch + 1) % 10 == 0 or epoch + 1 == epochs):
            model.to("cpu")
            iou, f = evaluate_pattern_model(model, val_ds)
            model.to(device)
            msg += f" | val IoU={iou:.4f} borderF={f:.4f}"
        bar.set_postfix_str(msg.split(' ', 2)[2])
        tqdm.write(msg)

    model.eval().to("cpu")
    torch.save({
        "arch": "PatternUNet", "input_size": size,
        "model_state_dict": model.state_dict(),
        "training_quality": {"epoch_losses": losses, "final_train_loss": losses[-1],
                             "best_train_loss": min(losses), "best_epoch": int(np.argmin(losses)) + 1,
                             "mean_last5_train_loss": float(np.mean(losses[-5:])),
                             "epochs": epochs, "dataset_size": len(train_ds)},
        "train_config": {"lr": lr, "batch_size": batch_size, "augment": augment, "size": size,
                         "loss": "BorderAwareBCEDiceLoss", "val_names": list(val_names or [])},
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }, model_path)
    print(f"✅ saved {model_path}")
    return model
