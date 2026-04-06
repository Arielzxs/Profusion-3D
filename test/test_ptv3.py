import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

sys.path.append("/root/autodl-tmp/Pointcept")
from pointcept.models.builder import build_model as pc_build_model
from pointcept.utils.config import Config

DATA_ROOT = "/root/autodl-tmp/nuscenes_mini"
CKPT = "/root/autodl-tmp/ptv3_ckpt.pth"  
CFG_PATH = "/root/autodl-tmp/Pointcept/configs/nuscenes/semseg-pt-v3m1-0-base.py"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    cfg = Config.fromfile(CFG_PATH)
    model = pc_build_model(cfg.model)
    state = torch.load(CKPT, map_location="cpu")
    state = state.get("state_dict", state)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()

    nusc = NuScenes(version="v1.0-mini", dataroot=DATA_ROOT, verbose=False)
    
    # 建立 16x16 的混淆矩阵 (只包含有效类 0-15)
    hist = np.zeros((16, 16), dtype=np.int64)
    grid_size = np.array([0.05, 0.05, 0.05], dtype=np.float32)

    for sample in tqdm(nusc.sample, desc="Evaluating native PTv3"):
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        pc = LidarPointCloud.from_file(os.path.join(DATA_ROOT, lidar_sd["filename"]))
        
        points = pc.points[:3].T.astype(np.float32)  
        
        # ====== 强度强度 ======
        # Pointcept 官方在 nuScenes 的强度上通常是保持 0-255 原始值的。
        # 如果你的提取出错了，你可以尝试改成 /255.0，但我们先按官方默认保持原始值
        intensity = np.zeros_like(pc.points[3:4].T.astype(np.float32))
        
        # 获取真实标签
        lidseg = nusc.get("lidarseg", lidar_sd["token"])
        raw_labels = np.fromfile(os.path.join(DATA_ROOT, lidseg["filename"]), dtype=np.uint8)
        
        # ====== 🚨 核心修复：官方的 learning_map ======
        # 官方配置中的 ignore_index 是 -1，而不是 0！
        # 有效类别是 0 到 15。
        map_dict = {1: -1, 5: -1, 7: -1, 8: -1, 10: -1, 11: -1, 13: -1, 19: -1, 20: -1, 0: -1, 
                    29: -1, 31: -1, 9: 0, 14: 1, 15: 2, 16: 2, 17: 3, 18: 4, 21: 5, 2: 6, 
                    3: 6, 4: 6, 6: 6, 12: 7, 22: 8, 23: 9, 24: 10, 25: 11, 26: 12, 
                    27: 13, 28: 14, 30: 15}
        learning_map = np.full(32, -1, dtype=np.int64) # 初始化为全 -1
        for k, v in map_dict.items(): learning_map[k] = v
        labels = learning_map[raw_labels]  # [N] 的值为 -1 到 15

        # 计算离散网格坐标
        grid_coord = np.floor(points / grid_size).astype(np.int32)
        unique_voxel_coords, unique_idx, inverse_map = np.unique(
            grid_coord, axis=0, return_index=True, return_inverse=True
        )
        
        voxel_points = points[unique_idx]
        voxel_intensity = intensity[unique_idx]
        M = len(unique_idx)
        
        # ====== 🚨 核心修复：添加 pdnorm 条件 ======
        input_dict = {
            "coord": torch.tensor(voxel_points, device=DEVICE),
            "grid_coord": torch.tensor(unique_voxel_coords, device=DEVICE),
            "offset": torch.tensor([M], device=DEVICE),
            "feat": torch.tensor(np.concatenate([voxel_points, voxel_intensity], axis=1), device=DEVICE),
            "batch": torch.zeros(M, dtype=torch.long, device=DEVICE),
            # 因为是多数据集模型，必须告诉它这是 nuScenes 数据！
            # 对应的 condition_idx 是 pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo") 中的索引
            "condition": torch.zeros(M, dtype=torch.long, device=DEVICE) 
        }

        with torch.no_grad():
            logits = model(input_dict)  
            
            if isinstance(logits, dict):
                logits = logits.get("seg_logits", logits.get("logits", logits.get("feat", logits)))

            full_logits = logits[inverse_map]
            
            # 模型输出形状是 16，直接预测 0-15
            pred = full_logits.argmax(1).cpu().numpy()
            gt = labels
            
            # 过滤 ignore_index == -1
            mask = gt != -1
            for g, p in zip(gt[mask], pred[mask]):
                if p < 16:
                    hist[g, p] += 1

    # 统计对角线预测对的数量
    diag = np.diag(hist)
    print("\n--- Diagnostic Stats ---")
    print(f"Total True Positives per class (0-15): {diag}")
    print(f"Total GT count per class (0-15): {hist.sum(1)}")
    
    # 计算 mIoU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-6)
    miou = float(np.nanmean(iu))
    
    print(f"\n🚀 Final Native PTv3 mIoU on mini val: {miou:.4f}")

if __name__ == "__main__":
    main()