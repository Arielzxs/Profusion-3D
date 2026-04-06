import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselinePTv3(nn.Module):
    """Baseline: 仅使用点特征的轻量分割头"""
    def __init__(self, ptv3_dim=16, hidden_dim=64, num_classes=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(ptv3_dim),
            nn.Linear(ptv3_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, point_features, **kwargs):
        return self.net(point_features)


class SimplePTv3Linear(nn.Module):
    """Simple: 点特征 -> MLP -> 分类"""
    def __init__(self, ptv3_dim=16, hidden_dim=128, num_classes=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(ptv3_dim),
            nn.Linear(ptv3_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, point_features, **kwargs):
        return self.net(point_features)


class DirectFusionModel(nn.Module):
    """
    Direct Fusion: 点特征 + 图像特征直接拼接，然后 MLP 分类
    需要外部先把图像特征投影到点上（见 train.py 中的适配）
    """
    def __init__(self, ptv3_dim=16, dinov2_dim=384, hidden_dim=64, num_classes=32):
        super().__init__()
        self.ln_pt = nn.LayerNorm(ptv3_dim)
        self.ln_img = nn.LayerNorm(dinov2_dim)
        self.mlp = nn.Sequential(
            nn.Linear(ptv3_dim + dinov2_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, point_features, image_features, **kwargs):
        pt = self.ln_pt(point_features)
        img = self.ln_img(image_features)
        fused = torch.cat([pt, img], dim=-1)
        return self.mlp(fused)


class ProjectionFusionPaper(nn.Module):
    """
    Projection Fusion（论文版）:
    - 将图像特征投影到点
    - 点/图特征分别线性变换后通过 gate 融合
    - 不使用 BEV 残差
    """
    def __init__(self, ptv3_dim=16, dinov2_dim=384, hidden_dim=128, num_classes=32):
        super().__init__()
        self.ln_pt = nn.LayerNorm(ptv3_dim)
        self.ln_img = nn.LayerNorm(dinov2_dim)
        self.pt_proj = nn.Linear(ptv3_dim, hidden_dim)
        self.img_proj = nn.Linear(dinov2_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        nn.init.zeros_(self.head.bias)

    def forward(self, point_features, image_features, **kwargs):
        pt = self.pt_proj(self.ln_pt(point_features))
        img = self.img_proj(self.ln_img(image_features))
        gate = self.gate(torch.cat([pt, img], dim=-1))
        fused = gate * img + (1 - gate) * pt
        logits = self.head(self.out_ln(fused))
        return logits

class ImprovedProjectionFusionModel(nn.Module):
    """
    保留你仓库原有的改进版 Projection Fusion（含 BEV 残差和 alpha）。
    原文件的 ProjectionFusionModel 代码整体移植到这里。
    """
    def __init__(self, ptv3_dim=16, dinov2_dim=384, hidden_dim=128, num_classes=32,
                 x_range=(-50, 50), y_range=(-50, 50), bev_res=0.5, alpha_init=0.0):
        super().__init__()
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.bev_res = bev_res
        self.num_classes = num_classes

        self.lidar_proj = nn.Sequential(
            nn.LayerNorm(ptv3_dim),
            nn.Linear(ptv3_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.img_proj = nn.Sequential(
            nn.LayerNorm(dinov2_dim),
            nn.Linear(dinov2_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.bev_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        )
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.seg_head = nn.Linear(hidden_dim, num_classes)
        nn.init.xavier_uniform_(self.seg_head.weight, gain=0.1)
        nn.init.zeros_(self.seg_head.bias)

    @torch.no_grad()
    def project_feat(self, voxel_coords, calib_matrices, img_feats, img_shape, feat_shape):
        device = voxel_coords.device
        V = voxel_coords.shape[0]
        cams, C, Hf, Wf = img_feats.shape
        Himg, Wimg = img_shape

        vis_acc = torch.zeros((V, C), device=device)
        vis_cnt = torch.zeros((V, 1), device=device)
        homo = torch.cat([voxel_coords, torch.ones((V, 1), device=device)], dim=1)

        for cam in range(cams):
            RT = calib_matrices[cam]
            proj = homo @ RT.t()
            u, v, w = proj[:, 0], proj[:, 1], proj[:, 2]
            u = u / w.clamp(min=1e-6)
            v = v / w.clamp(min=1e-6)

            u_norm = (u / (Wimg / 2)) - 1
            v_norm = (v / (Himg / 2)) - 1

            grid = torch.stack([u_norm, v_norm], dim=1).view(1, V, 1, 2)
            feat_map = img_feats[cam].unsqueeze(0)
            sampled = F.grid_sample(feat_map, grid, align_corners=True, mode="bilinear", padding_mode="zeros")
            sampled = sampled.squeeze(0).squeeze(-1).permute(1, 0)

            mask = (w > 0) & (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg)
            vis_acc[mask] += sampled[mask]
            vis_cnt[mask] += 1

        vis_cnt[vis_cnt == 0] = 1
        vis_feat = vis_acc / vis_cnt
        vis_mask = (vis_cnt > 0).float()
        return vis_feat, vis_mask

    def bev_scatter(self, feats, coords):
        feats = feats.float()
        x, y = coords[:, 0], coords[:, 1]
        Hbev = int((self.y_max - self.y_min) / self.bev_res)
        Wbev = int((self.x_max - self.x_min) / self.bev_res)
        ix = torch.floor((x - self.x_min) / self.bev_res).long()
        iy = torch.floor((y - self.y_min) / self.bev_res).long()
        mask = (ix >= 0) & (ix < Wbev) & (iy >= 0) & (iy < Hbev)

        idx_flat = iy * Wbev + ix
        idx_valid = idx_flat[mask]
        feats_valid = feats[mask]

        C = feats.shape[1]
        bev = torch.zeros((Hbev * Wbev, C), device=feats.device, dtype=feats.dtype)
        cnt = torch.zeros((Hbev * Wbev, 1), device=feats.device, dtype=feats.dtype)
        bev.index_add_(0, idx_valid, feats_valid)
        cnt.index_add_(0, idx_valid, torch.ones((idx_valid.size(0), 1), device=feats.device, dtype=feats.dtype))
        cnt[cnt == 0] = 1
        bev = (bev / cnt).view(Hbev, Wbev, C).permute(2, 0, 1).contiguous()
        return bev, idx_flat, mask, (Hbev, Wbev)

    def forward(self, point_features, image_features, voxel_coords, calib_matrices, img_shape, feat_shape):
        img_shape = img_shape.to(point_features.device)
        feat_shape = feat_shape.to(point_features.device)

        vis_feat, _ = self.project_feat(voxel_coords, calib_matrices, image_features, img_shape, feat_shape)

        lidar_emb = self.lidar_proj(point_features)
        img_emb   = self.img_proj(vis_feat)

        gate = self.gate(torch.cat([lidar_emb, img_emb], dim=-1))
        fused_pt = gate * img_emb + (1 - gate) * lidar_emb  # 点级主通路

        # BEV 残差
        bev_lidar, idx_flat, mask_inside, _ = self.bev_scatter(fused_pt, voxel_coords)
        bev_img, _, _, _ = self.bev_scatter(img_emb, voxel_coords)
        bev_fused = bev_lidar + 0.5 * self.bev_fuse(bev_img)
        bev_flat = bev_fused.permute(1, 2, 0).reshape(-1, bev_fused.shape[0]).to(fused_pt.dtype)
        pts_feat = fused_pt.clone()
        if self.alpha.item() != 0:
            pts_feat[mask_inside] = fused_pt[mask_inside] + torch.clamp(self.alpha, 0.0, 1.0) * bev_flat[idx_flat[mask_inside]]

        pts_feat = self.out_ln(pts_feat)
        logits = self.seg_head(pts_feat)
        return logits

def build_model(name: str, ptv3_dim=16, dinov2_dim=384, hidden_dim=128,
                num_classes=32, alpha=0.0):
    name = name.lower()
    if name == "baseline":
        return BaselinePTv3(ptv3_dim, hidden_dim=64, num_classes=num_classes)
    if name == "simple":
        return SimplePTv3Linear(ptv3_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    if name == "direct":
        return DirectFusionModel(ptv3_dim, dinov2_dim, hidden_dim, num_classes)
    if name == "projection":
        return ProjectionFusionPaper(ptv3_dim, dinov2_dim, hidden_dim, num_classes)
    if name == "improved":
        return ImprovedProjectionFusionModel(ptv3_dim, dinov2_dim, hidden_dim, num_classes,
                                             alpha_init=alpha)
    raise ValueError(f"Unknown model type: {name}")