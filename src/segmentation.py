import torch


class EdgeAwareLoss(torch.nn.Module):
    """
    Edge-aware loss combining BCE, Dice, Compactness and boundary penalties.

    This loss adds two edge-specific terms:
    - Boundary loss: penalizes errors specifically at object boundaries
    - Edge gradient loss: ensures predicted edges match target edge gradients

    Parameters
    ----------
    smooth : float, optional
        Smoothing factor to avoid division by zero, by default 1e-6
    compact_weight : float, optional
        Weight for compactness loss term, by default 0.2
    edge_weight : float, optional
        Weight for edge penalty term, by default 0.5
    boundary_dilation : int, optional
        Number of pixels to dilate boundary mask, by default 3

    References
    ----------
    - Boundary loss: https://arxiv.org/abs/1905.07852
    - Surface loss for highly unbalanced segmentation: MIDL 2019
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        compact_weight: float = 0.2,
        edge_weight: float = 0.5,
        boundary_dilation: int = 3,
    ):
        super().__init__()
        self.bce = torch.nn.BCELoss()
        self.smooth = smooth
        self.edge_weight = edge_weight
        self.boundary_dilation = boundary_dilation

        # Sobel kernels for gradient computation
        self._register_sobel_kernels()

    def _register_sobel_kernels(self) -> None:
        """Register Sobel filter kernels as non-trainable buffers."""
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).reshape(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).reshape(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _compute_edges(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute edge map from a binary mask using Sobel filters.

        Parameters
        ----------
        mask : torch.Tensor
            Binary mask of shape (B, 1, H, W)

        Returns
        -------
        torch.Tensor
            Edge magnitude map of shape (B, 1, H, W), values in [0, 1]
        """
        grad_x = torch.nn.functional.conv2d(mask, self.sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(mask, self.sobel_y, padding=1)
        edges = torch.sqrt(grad_x**2 + grad_y**2 + self.smooth)
        # Normalize to [0, 1]
        return torch.clamp(edges / (edges.max() + self.smooth), 0, 1)

    def _get_boundary_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Generate dilated boundary mask from ground truth.

        Dilation ensures the model is penalized in a neighborhood
        around the true boundary, not just on exact pixels.

        Parameters
        ----------
        targets : torch.Tensor
            Ground truth mask of shape (B, 1, H, W)

        Returns
        -------
        torch.Tensor
            Binary boundary mask of shape (B, 1, H, W)
        """
        kernel_size = 2 * self.boundary_dilation + 1
        kernel = torch.ones(
            1, 1, kernel_size, kernel_size, device=targets.device
        )

        # Erode and dilate to isolate boundary region
        dilated = torch.nn.functional.conv2d(
            targets, kernel, padding=self.boundary_dilation
        ).clamp(0, 1)

        eroded = 1 - torch.nn.functional.conv2d(
            1 - targets, kernel, padding=self.boundary_dilation
        ).clamp(0, 1)

        return (dilated - eroded).clamp(0, 1)

    def _dice_loss(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute standard Dice loss.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted probabilities
        targets : torch.Tensor
            Ground truth binary mask

        Returns
        -------
        torch.Tensor
            Scalar Dice loss
        """
        intersection = (preds * targets).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )

    def _edge_loss(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge-specific loss combining gradient and boundary terms.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted probabilities of shape (B, 1, H, W)
        targets : torch.Tensor
            Ground truth mask of shape (B, 1, H, W)

        Returns
        -------
        torch.Tensor
            Scalar edge loss
        """
        pred_edges = self._compute_edges(preds)
        target_edges = self._compute_edges(targets)

        # L1 gradient matching on edges
        gradient_loss = torch.nn.functional.l1_loss(pred_edges, target_edges)

        # BCE weighted by boundary mask — harder penalty near boundaries
        boundary_mask = self._get_boundary_mask(targets)
        boundary_bce = torch.nn.functional.binary_cross_entropy(
            preds * boundary_mask, targets * boundary_mask, reduction="sum"
        ) / (boundary_mask.sum() + self.smooth)

        return gradient_loss + boundary_bce

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined edge-aware loss.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted probabilities of shape (B, 1, H, W)
        targets : torch.Tensor
            Ground truth binary mask of shape (B, 1, H, W)

        Returns
        -------
        torch.Tensor
            Combined scalar loss value
        """
        bce_loss = self.bce(preds, targets)
        dice_loss = self._dice_loss(preds, targets)
        edge_loss = self._edge_loss(preds, targets)

        return bce_loss + dice_loss + self.edge_weight * edge_loss
