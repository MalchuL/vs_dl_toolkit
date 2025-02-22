import torch
from torch import nn
import torch.nn.functional as F


class StructureLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    @staticmethod
    def attn_cosine_sim(x, eps=1e-08):
        x = x / torch.clamp(x.norm(dim=1, keepdim=True), min=eps)
        sim_matrix = torch.matmul(x, x.transpose(-1, -2))
        return sim_matrix

    def forward(self, pred, target):
        sim_pred = self.attn_cosine_sim(pred)
        with torch.no_grad():
            sim_target = self.attn_cosine_sim(target)
        return F.mse_loss(sim_pred, sim_target, reduction=self.reduction)
