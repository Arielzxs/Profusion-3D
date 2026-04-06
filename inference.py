import torch
from models import ImprovedProjectionFusionModel  # 或 ProjectionFusionPaper

def run_one(sample_pt="/root/autodl-tmp/data_features/layer_12/val/any_sample.pt",
            ckpt="best_fusion_model.pth",
            model_type="improved",  # "improved" 或 "projection"（论文版）
            num_classes=32, ptv3_dim=16, dinov2_dim=384, hidden_dim=128):
    data = torch.load(sample_pt, map_location="cpu")
    if model_type == "improved":
        model = ImprovedProjectionFusionModel(
            ptv3_dim=ptv3_dim, dinov2_dim=dinov2_dim,
            hidden_dim=hidden_dim, num_classes=num_classes)
        with torch.no_grad():
            logits = model(
                data["point_features"],
                data["image_features"].permute(0, 3, 1, 2),  # [6,C,Hf,Wf]
                data["voxel_coords"],
                data["calib_matrices"],
                data["img_shape"],
                data["feat_shape"],
            )
    else:  # "projection" -> ProjectionFusionPaper 接受 pt + vis
        from models import ProjectionFusionPaper
        model = ProjectionFusionPaper(
            ptv3_dim=ptv3_dim, dinov2_dim=dinov2_dim,
            hidden_dim=hidden_dim, num_classes=num_classes)
        # 生成 vis 特征（与 train 中一致）
        from train.train import project_img_to_points
        vis = project_img_to_points(
            data["voxel_coords"],
            data["calib_matrices"],
            data["image_features"].permute(0, 3, 1, 2),
            data["img_shape"],
        )
        with torch.no_grad():
            logits = model(data["point_features"], vis)

    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model.eval()
    pred = logits.argmax(dim=1)
    print("Pred shape:", pred.shape)
    return pred

if __name__ == "__main__":
    run_one()