import torch
import torch.nn.functional as F

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
        return self.weight * (dh.mean() + dw.mean())