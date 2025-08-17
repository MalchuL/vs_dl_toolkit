import torch
import torch.nn.functional as F

from dl_toolkit.modules.toolkit_module import ToolkitModule


class StructureLoss(ToolkitModule):
    def __init__(self, reduction: str = "mean", eps=1e-08):
        """Structure loss.
        Structure loss is a regularization term that penalizes the structure of the image.
        It is used to prevent the model from overfitting to the input data.
        It is also used to improve the quality of the generated image.
        It is also used to improve the quality of the generated image.
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    @staticmethod
    def attn_cosine_sim(x, eps=1e-08):
        x = x / torch.clamp(x.norm(dim=1, keepdim=True), min=eps)
        sim_matrix = torch.matmul(x, x.transpose(-1, -2))
        return sim_matrix

    def forward(self, pred, target):
        sim_pred = self.attn_cosine_sim(pred, eps=self.eps)
        with torch.no_grad():
            sim_target = self.attn_cosine_sim(target, eps=self.eps)
        return F.mse_loss(sim_pred, sim_target, reduction=self.reduction)
