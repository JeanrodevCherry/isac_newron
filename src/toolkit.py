import cv2
import numpy as np
import torch
def build_overlay(image_gray, mask_prob, threshold=0.5, alpha=0.45):
    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    mask_bin = (mask_prob > threshold).astype(np.uint8)

    colored = image_bgr.copy()
    colored[mask_bin > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(colored, alpha, image_bgr, 1.0 - alpha, 0)

    mask_vis = (mask_bin * 255).astype(np.uint8)
    return overlay, mask_vis


def pattern_region(pattern_model, img, size=None, threshold=0.5, expand_ratio=1.15):
    """Run the pattern (denoising) model on a grayscale image.

    Keeps only the largest connected region of the pattern mask and returns
    (pattern_mask, (x1, y1, x2, y2), pattern_prob) at the original resolution,
    or (None, None, prob) if nothing is detected. The box is a square crop
    around the pattern.
    """
    h, w = img.shape
    size = size or getattr(pattern_model, "input_size", 256)
    inp = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    device = next(pattern_model.parameters()).device
    inp_t = (torch.from_numpy(inp).float()[None, None] / 255.0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(pattern_model(inp_t))[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    binary = (prob > threshold).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None, None, prob
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == k).astype(np.uint8)
    bw, bh = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
    cx, cy = int(centroids[k][0]), int(centroids[k][1])
    r = int(max(bw, bh) * expand_ratio / 2)
    box = (max(0, cx - r), max(0, cy - r), min(w, cx + r), min(h, cy + r))
    return mask, box, prob
