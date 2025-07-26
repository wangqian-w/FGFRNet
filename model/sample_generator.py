import torch
import torch.nn as nn
import torch.nn.functional as F

class SampleGenerator(nn.Module):
    def __init__(self, max_samples=256):
        super(SampleGenerator, self).__init__()

        self.max_samples = max_samples
        self.relu = nn.ReLU(inplace=True)
        self.projector = nn.Sequential(
            nn.Conv2d(16, 32, 3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
        )

    def generate(self, F_feat, Y_mask, T_img):

        B, C, Hf, Wf = F_feat.shape

        residual=F_feat
        F_feat_mlp = self.relu(self.projector(F_feat)+residual)  # [B, C, Hf, Wf]

        F_fore_all, F_back_all = [], []
        M_fore_all, M_back_all = [], []
        T_fore_all, T_back_all = [], []

        for b in range(B):

            feat = F_feat[b]  # [C, Hf, Wf]
            feat_mlp = F_feat_mlp[b]  # [C, Hf, Wf]
            mask = Y_mask[b, 0]  # [Hf, Wf]
            temp = T_img[b, 0]  # [Hf, Wf]

            feat = feat.view(C, -1).T  # [Hf*Wf, C]
            feat_mlp = feat_mlp.view(C, -1).T  # [Hf*Wf, C]
            mask = mask.view(-1)  # [Hf*Wf]
            temp = temp.view(-1)  # [Hf*Wf]

            M_fore_idx = (mask == 1).nonzero(as_tuple=False).squeeze(1)
            M_back_idx = (mask == 0).nonzero(as_tuple=False).squeeze(1)

            if M_fore_idx.numel() < 2:
                M_fore_idx = torch.arange(len(mask), device=mask.device)
            if M_back_idx.numel() < 1:
                M_back_idx = torch.arange(len(mask), device=mask.device)

            min_fore_samples = min(len(M_fore_idx), self.max_samples)
            min_back_samples = min(len(M_back_idx), self.max_samples)

            M_fore_idx = M_fore_idx[torch.randperm(len(M_fore_idx))[:min_fore_samples]]
            M_back_idx = M_back_idx[torch.randperm(len(M_back_idx))[:min_back_samples]]

            F_fore_all.append(F.normalize(feat_mlp[M_fore_idx], dim=1))
            F_back_all.append(F.normalize(feat_mlp[M_back_idx], dim=1))

            M_fore_all.append(F.normalize(feat[M_fore_idx], dim=1))
            M_back_all.append(F.normalize(feat[M_back_idx], dim=1))

            T_fore_all.append(temp[M_fore_idx])
            T_back_all.append(temp[M_back_idx])

        return (
            F_feat_mlp,
            torch.cat(F_fore_all, dim=0),
            torch.cat(F_back_all, dim=0),
            torch.cat(M_fore_all, dim=0),
            torch.cat(M_back_all, dim=0),
            torch.cat(T_fore_all, dim=0),
            torch.cat(T_back_all, dim=0)
        )

    def forward(self, F_feat, Y_mask, T_img):
        return self.generate(F_feat, Y_mask, T_img)