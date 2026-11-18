import json
import os
from pathlib import Path

import cv2
import numpy as np

root = Path(r"C:\Users\cherry\Documents\dev\isac_newron\data\segmentation\images")
temp = Path(os.environ["TEMP"])
names = [f"D00_Plate_1_D{i}_ch00_SV" for i in range(1, 7)]
panels = []
for stem in names:
    image = cv2.imread(str(root / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
    before = json.loads((temp / f"{stem}.json.before_edge_refinement").read_text(encoding="utf-8"))
    after = json.loads((root / f"{stem}.json").read_text(encoding="utf-8"))
    left = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    right = left.copy()
    for shape in before["shapes"]:
        if shape.get("shape_type") == "polygon":
            pts = np.round(shape["points"]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(left, [pts], True, (0, 220, 0), 3, cv2.LINE_AA)
    for shape in after["shapes"]:
        if shape.get("shape_type") == "polygon":
            pts = np.round(shape["points"]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(right, [pts], True, (0, 0, 255), 3, cv2.LINE_AA)
    scale = 0.26
    left = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    label = stem.replace("D00_Plate_1_", "")
    cv2.putText(left, "BEFORE (green)", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)
    cv2.putText(right, "AFTER (red)", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    pair = np.hstack([left, right])
    cv2.putText(pair, label, (15, pair.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)
    panels.append(pair)
panel_width = max(panel.shape[1] for panel in panels)
panel_height = max(panel.shape[0] for panel in panels)
padded = [cv2.copyMakeBorder(panel, 0, panel_height - panel.shape[0], 0, panel_width - panel.shape[1], cv2.BORDER_CONSTANT, value=(32, 32, 32)) for panel in panels]
rows = [np.hstack(padded[i:i + 2]) for i in (0, 2, 4)]
canvas = np.vstack(rows)
cv2.imwrite(str(temp / "edge_refinement_before_after.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
