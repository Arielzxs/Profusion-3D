import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from models import build_model, ImprovedProjectionFusionModel
from dataset import FeatureDataset, collate_list
import lovasz_losses
import argparse, os, csv
import numpy as np

@torch.no_grad()
def project_img_to_points(voxel_coords, calib_matrices, img_feats, img_shape):
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
        u = u / w.clamp(min=1e-6); v = v / w.clamp(min=1e-6)
        u_norm = (u / (Wimg / 2)) - 1
        v_norm = (v / (Himg / 2)) - 1
        grid = torch.stack([u_norm, v_norm], dim=1).view(1, V, 1, 2)
        feat_map = img_feats[cam].unsqueeze(0)
        sampled = torch.nn.functional.grid_sample(
            feat_map, grid, align_corners=True, mode="bilinear", padding_mode="zeros"
        )
        sampled = sampled.squeeze(0).squeeze(-1).permute(1, 0)
        mask = (w > 0) & (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg)
        vis_acc[mask] += sampled[mask]; vis_cnt[mask] += 1
    vis_cnt[vis_cnt == 0] = 1
    vis_feat = vis_acc / vis_cnt
    return vis_feat

def compute_metrics(model, dl, device, num_classes):
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for batch in dl:
            for sample in batch:
                pt  = sample["point_features"].to(device)
                img = sample["image_features"].to(device)
                xyz = sample["voxel_coords"].to(device)
                cal = sample["calib_matrices"].to(device)
                img_shape = sample["img_shape"].to(device)
                feat_shape = sample["feat_shape"].to(device)
                lbl = sample["labels"].to(device).long()

                cls_name = model.__class__.__name__
                if cls_name == "DirectFusionModel":
                    vis = project_img_to_points(xyz, cal, img, img_shape)
                    logits = model(pt, vis)
                elif cls_name in ["ImprovedProjectionFusionModel"]:
                    logits = model(pt, img, xyz, cal, img_shape, feat_shape)
                elif cls_name in ["ProjectionFusionPaper"]:
                    vis = project_img_to_points(xyz, cal, img, img_shape)
                    logits = model(pt, vis)
                else:  # Baseline / Simple
                    logits = model(pt)

                pred = logits.argmax(1).cpu().numpy()
                gt   = lbl.cpu().numpy()
                mask = gt != 0
                for g, p in zip(gt[mask], pred[mask]):
                    hist[g, p] += 1

    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-6)
    miou = float(np.nanmean(iu[1:]))
    acc = np.diag(hist) / (hist.sum(1) + 1e-6)
    macc = float(np.nanmean(acc[1:]))
    freq = hist.sum(1) / (hist.sum() + 1e-6)
    fwiou = float(np.nansum(freq * iu))
    return miou, macc, fwiou

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = FeatureDataset(args.train_dir)
    val_ds   = FeatureDataset(args.val_dir)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_list)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_list)

    model = build_model(
        name=args.model_type,
        ptv3_dim=args.ptv3_dim,
        dinov2_dim=args.dinov2_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        alpha=args.alpha
    ).to(device)

    # Linear head 需要较大的学习率，因此这里默认给 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # 丢弃自定义 weight，直接用纯净的 CE Loss
    ce = nn.CrossEntropyLoss(ignore_index=0)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)
    with open(args.log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "miou", "macc", "fwiou", "lr", "model"])

    best = -1.0
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for batch in train_dl:
            optimizer.zero_grad()
            batch_loss_sum = 0.0

            for sample in batch:
                with autocast(enabled=torch.cuda.is_available()):
                    pt  = sample["point_features"].to(device)
                    img = sample["image_features"].to(device)
                    xyz = sample["voxel_coords"].to(device)
                    cal = sample["calib_matrices"].to(device)
                    img_shape = sample["img_shape"].to(device)
                    feat_shape = sample["feat_shape"].to(device)
                    lbl = sample["labels"].to(device).long()

                    cls_name = model.__class__.__name__
                    if cls_name == "DirectFusionModel":
                        vis = project_img_to_points(xyz, cal, img, img_shape)
                        logits = model(pt, vis)
                    elif cls_name in ["ImprovedProjectionFusionModel"]:
                        logits = model(pt, img, xyz, cal, img_shape, feat_shape)
                    elif cls_name in ["ProjectionFusionPaper"]:
                        vis = project_img_to_points(xyz, cal, img, img_shape)
                        logits = model(pt, vis)
                    else:  # Baseline / Simple
                        logits = model(pt)

                    loss_ce = ce(logits, lbl)
                    probs = torch.softmax(logits, dim=1)
                    # 提高 lovasz 权重的占比来对抗类别不平衡，这是业界标准做法
                    loss_lv = lovasz_losses.lovasz_softmax(probs, lbl, ignore=0)
                    loss = (loss_ce + 1.0 * loss_lv) / len(batch)

                scaler.scale(loss).backward()
                batch_loss_sum += loss.item() * len(batch)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += batch_loss_sum

        scheduler.step()
        epoch_loss = running / len(train_ds)
        miou, macc, fwiou = compute_metrics(model, val_dl, device, args.num_classes)
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{args.epochs} | Loss {epoch_loss:.4f} | mIoU {miou:.4f} | mAcc {macc:.4f} | fwIoU {fwiou:.4f} | lr {lr_now:.6f}")
        with open(args.log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, epoch_loss, miou, macc, fwiou, lr_now, args.model_type])

        if miou > best:
            best = miou
            torch.save(model.state_dict(), args.ckpt)
            print(f"  -> best val mIoU={miou:.4f} saved to {args.ckpt}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["baseline", "simple", "direct", "projection", "improved"],
                   default="simple", help="论文模型与改进模型选择")
    p.add_argument("--dino_layer", type=int, choices=[9, 12], default=9)
    p.add_argument("--train_dir", default="/root/autodl-tmp/data_features/layer_9/train")
    p.add_argument("--val_dir",   default="/root/autodl-tmp/data_features/layer_9/val")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_classes", type=int, default=17)
    p.add_argument("--ptv3_dim", type=int, default=64)
    p.add_argument("--dinov2_dim", type=int, default=384)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--ckpt", default="best_fusion_model.pth")
    p.add_argument("--log_path", default="logs/loss.csv")
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--alpha", type=float, default=0.0)
    args = p.parse_args()
    train(args)