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


class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()

        tp = (pred * target).sum(dim=(1, 2, 3))
        fp = (pred * (1.0 - target)).sum(dim=(1, 2, 3))
        fn = ((1.0 - pred) * target).sum(dim=(1, 2, 3))

        score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - score.mean()


class EdgeConsistencyLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def _grad_mag(self, tensor):
        gx = F.conv2d(tensor, self.kernel_x, padding=1)
        gy = F.conv2d(tensor, self.kernel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + self.eps)

    def forward(self, pred, target):
        edge_pred = self._grad_mag(pred)
        edge_target = self._grad_mag(target)
        return F.l1_loss(edge_pred, edge_target)


class NoInnerHoleLoss(torch.nn.Module):
    """
    Penalizes likely inner holes by comparing prediction against its own soft-closing.
    The penalty is applied only where ground-truth is foreground.
    """
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, pred, target):
        pad = self.kernel_size // 2

        # Soft closing = erosion(dilation(pred))
        dilated = F.max_pool2d(pred, kernel_size=self.kernel_size, stride=1, padding=pad)
        closed = -F.max_pool2d(-dilated, kernel_size=self.kernel_size, stride=1, padding=pad)

        # Missing interior regions are where closed > pred
        missing = torch.relu(closed - pred)
        inside_fg = missing * target
        return inside_fg.mean()


class BCETverskyBoundaryNoHoleLoss(torch.nn.Module):
    def __init__(
        self,
        tversky_weight=1.0,
        boundary_weight=1.0,
        no_hole_weight=1.5,
    ):
        super().__init__()
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.no_hole_weight = no_hole_weight

        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.boundary = EdgeConsistencyLoss()
        self.no_hole = NoInnerHoleLoss(kernel_size=5)

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        tversky = self.tversky(probs, target)
        boundary = self.boundary(probs, target)
        no_hole = self.no_hole(probs, target)

        return (
            bce
            + self.tversky_weight * tversky
            + self.boundary_weight * boundary
            + self.no_hole_weight * no_hole
        )