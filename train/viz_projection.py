import os
import argparse
import random
import torch
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import matplotlib.pyplot as plt
from models import ImprovedProjectionFusionModel  # 若要论文版可改为 ProjectionFusionPaper

# 17类可视化色表（index 0~16）
COLOR_MAP_17 = np.array([
    [0, 0, 0],        # 0 ignore
    [255, 0, 0],      # 1
    [0, 255, 0],      # 2
    [0, 0, 255],      # 3
    [255, 255, 0],    # 4
    [255, 0, 255],    # 5
    [0, 255, 255],    # 6
    [255, 128, 0],    # 7
    [128, 0, 255],    # 8
    [0, 128, 255],    # 9
    [255, 128, 128],  # 10
    [128, 255, 128],  # 11
    [128, 128, 255],  # 12
    [200, 200, 0],    # 13
    [200, 0, 200],    # 14
    [0, 200, 200],    # 15
    [180, 120, 0],    # 16
], dtype=np.uint8)


def build_learning_map():
    # nuScenes lidarseg(0~31) -> 17类
    map_dict = {
        1: 0, 5: 0, 7: 0, 8: 0, 10: 0, 11: 0, 13: 0, 19: 0, 20: 0, 0: 0,
        29: 0, 31: 0, 9: 1, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 21: 6, 2: 7,
        3: 7, 4: 7, 6: 7, 12: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
        27: 14, 28: 15, 30: 16
    }
    lm = np.zeros(32, dtype=np.int64)
    for k, v in map_dict.items():
        lm[k] = v
    return lm


def load_sample(nusc, token, data_root, learning_map):
    sample = nusc.get("sample", token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pc = LidarPointCloud.from_file(os.path.join(data_root, lidar_sd["filename"]))
    points = torch.tensor(pc.points.T, dtype=torch.float32)  # [N,4]

    try:
        lidseg = nusc.get("lidarseg", lidar_sd["token"])
        raw_labels = np.fromfile(os.path.join(data_root, lidseg["filename"]), dtype=np.uint8)
        labels = learning_map[raw_labels]  # 17类
    except Exception:
        labels = np.zeros(points.shape[0], dtype=np.int64)

    return points[:, :3], torch.tensor(labels, dtype=torch.long), sample


def project_points(xyz, cam_intr_extr, img_w, img_h):
    # xyz: [N,3], cam_intr_extr: [3,4]，同设备
    V = xyz.shape[0]
    homo = torch.cat([xyz, torch.ones((V, 1), device=xyz.device, dtype=xyz.dtype)], dim=1)  # [N,4]
    proj = homo @ cam_intr_extr.t()  # [N,3]
    u, v, w = proj[:, 0], proj[:, 1], proj[:, 2]
    u = u / w.clamp(min=1e-6)
    v = v / w.clamp(min=1e-6)

    mask = (w > 0) & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return u[mask].detach().cpu().numpy(), v[mask].detach().cpu().numpy(), mask


def draw_overlay(img_path, u, v, labels_np, title, point_radius=3, alpha=0.8):
    """
    把每个点渲染成小方块，避免点太小看不清
    point_radius=3 -> 方块大小 7x7
    """
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).astype(np.float32)

    labels_np = labels_np.astype(np.int64)
    labels_np = np.clip(labels_np, 0, COLOR_MAP_17.shape[0] - 1)
    colors = COLOR_MAP_17[labels_np].astype(np.float32)

    H, W = arr.shape[:2]
    uu = np.clip(u.astype(np.int32), 0, W - 1)
    vv = np.clip(v.astype(np.int32), 0, H - 1)

    for x, y, c in zip(uu, vv, colors):
        x0, x1 = max(0, x - point_radius), min(W, x + point_radius + 1)
        y0, y1 = max(0, y - point_radius), min(H, y + point_radius + 1)
        arr[y0:y1, x0:x1] = (1 - alpha) * arr[y0:y1, x0:x1] + alpha * c

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    plt.imshow(arr)
    plt.axis("off")
    plt.title(title)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nusc = NuScenes(version="v1.0-mini", dataroot=args.data_root, verbose=False)
    learning_map = build_learning_map()

    model = ImprovedProjectionFusionModel(
        ptv3_dim=args.ptv3_dim,
        dinov2_dim=args.dinov2_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    # token优先；否则从 data_dir 中随机取一个，保证文件存在
    if args.token is not None:
        token = args.token
    else:
        files = [f for f in os.listdir(args.data_dir) if f.endswith(".pt")]
        if len(files) == 0:
            raise FileNotFoundError(f"No .pt files in {args.data_dir}")
        token = random.choice(files).replace(".pt", "")

    feat_path = os.path.join(args.data_dir, f"{token}.pt")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature file not found: {feat_path}")

    xyz, gt_labels, sample = load_sample(nusc, token, args.data_root, learning_map)
    feat = torch.load(feat_path, map_location="cpu")

    pt_feat = feat["point_features"].to(device)  # [N,Cp]

    img_feat = feat["image_features"]
    if img_feat.dim() == 4 and img_feat.shape[1] != args.dinov2_dim:
        # [6,H,W,C] -> [6,C,H,W]
        img_feat = img_feat.permute(0, 3, 1, 2).contiguous()
    img_feat = img_feat.to(device)

    calib = feat["calib_matrices"].to(device)       # [6,3,4]
    img_shape = feat["img_shape"].to(device)        # [2] H,W
    feat_shape = feat["feat_shape"].to(device)      # [2] Hf,Wf (模型接口需要)

    xyz = xyz.to(device)
    gt_labels = gt_labels.to(device)

    with torch.no_grad():
        logits = model(pt_feat, img_feat, xyz, calib, img_shape, feat_shape)
        pred_labels = logits.argmax(1)  # [N]

    cam_token = sample["data"]["CAM_FRONT"]
    cam_sd = nusc.get("sample_data", cam_token)
    img_path = os.path.join(args.data_root, cam_sd["filename"])
    cam_idx = 0

    Himg = int(img_shape[0].item())
    Wimg = int(img_shape[1].item())

    u_gt, v_gt, mask_gt = project_points(xyz, calib[cam_idx], Wimg, Himg)
    u_pd, v_pd, mask_pd = project_points(xyz, calib[cam_idx], Wimg, Himg)

    gt_np = gt_labels[mask_gt].detach().cpu().numpy()
    pd_np = pred_labels[mask_pd].detach().cpu().numpy()

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    draw_overlay(
        img_path, u_gt, v_gt, gt_np,
        f"{token} GT(17cls)",
        point_radius=args.point_radius,
        alpha=args.alpha
    )
    plt.subplot(1, 2, 2)
    draw_overlay(
        img_path, u_pd, v_pd, pd_np,
        f"{token} Pred",
        point_radius=args.point_radius,
        alpha=args.alpha
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"{token}_viz_r{args.point_radius}_a{args.alpha:.2f}.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=220)
    print(f"[saved] {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/root/autodl-tmp/nuscenes_mini")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/data_features/layer_12/val")
    ap.add_argument("--ckpt", default="ckpt_improved_l12_a0_e40.pth")
    ap.add_argument("--out_dir", default="viz_out")
    ap.add_argument("--token", default=None)

    ap.add_argument("--num_classes", type=int, default=17)
    ap.add_argument("--ptv3_dim", type=int, default=64)
    ap.add_argument("--dinov2_dim", type=int, default=384)
    ap.add_argument("--hidden_dim", type=int, default=128)

    # 可视化增强参数
    ap.add_argument("--point_radius", type=int, default=3, help="点半径(像素)")
    ap.add_argument("--alpha", type=float, default=0.8, help="颜色覆盖透明度[0,1]")

    args = ap.parse_args()
    main(args)