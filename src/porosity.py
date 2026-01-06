import torch
import torch.nn.functional as F
import numpy as np

class CompactnessLoss(torch.nn.Module):
    """
    Penalizes strong gradients in the predicted mask -> discourages holes.
    Equivalent to total variation loss.
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred):
        # horizontal & vertical gradients
        dh = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        dw = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        area = torch.sum(pred)
        perimeter = self._calculate_perimeter(pred)
        return self.weight * (dh.mean() + dw.mean())

    def _calculate_perimeter(self, binary):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        grad_x = F.conv2d(binary, sobel_x.to(binary.device), padding=1)
        grad_y = F.conv2d(binary, sobel_y.to(binary.device), padding=1)
        edges = torch.sqrt(grad_x**2 + grad_y**2)
        return torch.sum(edges > 0.5).float()