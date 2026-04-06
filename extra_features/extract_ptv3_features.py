import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

# 让 Python 能找到 Pointcept 源码
sys.path.append("/root/autodl-tmp/Pointcept")
from pointcept.models.builder import build_model as pc_build_model
from pointcept.utils.config import Config
import pointcept.utils.comm as comm

DATA_ROOT = "/root/autodl-tmp/nuscenes_mini"
CKPT = "/root/autodl-tmp/ptv3_ckpt.pth"  # PTv3 权重
CFG_PATH = "/root/autodl-tmp/Pointcept/configs/nuscenes/semseg-pt-v3m1-0-base.py" # PTv3 配置文件
SAVE_DIR = "/root/autodl-tmp/ptv3_features"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##  构建模型并加载权重
def build_model(cfg_path, ckpt_path):
    cfg = Config.fromfile(cfg_path)
    model = pc_build_model(cfg.model)

    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("state_dict", state)
    # 去掉可能的 DataParallel 包装
    if any(k.startswith("module.") for k in state.keys()): 
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("Missing keys:", missing) # 模型中有但是state里没有的参数
        print("Unexpected keys:", unexpected) # state里有但是模型中没有的参数

    #将模型放到 GPU 上并设置为评估模式
    model.to(DEVICE).eval() 
    return model, cfg

def main():
    assert os.path.exists(CKPT), "请先把 PTv3 权重放到 CKPT 路径"
    assert os.path.exists(CFG_PATH), f"配置文件不存在: {CFG_PATH}"
    nusc = NuScenes(version="v1.0-mini", dataroot=DATA_ROOT, verbose=False)
    model, cfg = build_model(CFG_PATH, CKPT)

    # 直接硬编码 nuScenes 标准体素大小，绕过 cfg 解析错误
    grid_size_tensor = torch.tensor([0.05, 0.05, 0.05], device=DEVICE, dtype=torch.float32)

    for sample in tqdm(nusc.sample, desc="PTv3 feat"):
        token = sample["token"]
        out_path = os.path.join(SAVE_DIR, f"{token}.pt")
        if os.path.exists(out_path):
            continue

        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

        # 读取点云数据 [N, 4] (x, y, z, intensity)
        pc = LidarPointCloud.from_file(os.path.join(DATA_ROOT, lidar_sd["filename"])) 
        
        # [N, 3] 原始点云
        points = torch.tensor(pc.points[:3].T, dtype=torch.float32, device=DEVICE)
        # [N, 1] 强度
        intensity = torch.tensor(pc.points[3], dtype=torch.float32, device=DEVICE).unsqueeze(1)
        
        N_original = points.shape[0]

        # 计算每个点所处的整数网格坐标
        grid_coord = torch.floor(points / grid_size_tensor).int()
        
        # 找到唯一的体素，并记录原始点到体素的逆映射 inverse_map
        unique_coords, inverse_map = torch.unique(grid_coord, return_inverse=True, dim=0)
        
        M_voxels = unique_coords.shape[0]

        feat = torch.cat([points, intensity], dim=1)  # [N, 4]
        offset = torch.tensor([N_original], device=DEVICE)

        input_dict = {
            "coord": points,
            "grid_coord": grid_coord,  # 极其重要！网络底层靠它做稀疏卷积
            "feat": feat,
            "offset": offset, # 偏移量，告诉网络这是一个 batch 中的第一个样本（如果有多个样本，这里需要累加）
            "batch": torch.zeros(N_original, dtype=torch.long, device=DEVICE),
            "inverse": inverse_map  # 提供给可能需要它的网络层
        }

        with torch.no_grad():
            # 大多数 Segmentor 支持完整的 forward，能直接插值回 N 个点
            out = model(input_dict)
            
            # 如果输出已经是字典，且包含了最终上采样后的对齐特征
            if isinstance(out, dict) and "feat" in out:
                pt_feat = out["feat"]
            elif hasattr(out, "feat"):
                pt_feat = out.feat
            else:
                # 极端情况 fallback：调用 backbone，拿到 M 个 voxel 的特征，手动逆映射回 N 个点
                out = model.backbone(input_dict)
                feat_out = out.feat if hasattr(out, "feat") else out["feat"]
                
                if feat_out.shape[0] == M_voxels:
                    # 说明这是 Voxel 特征，手动逆映射！
                    pt_feat = feat_out[inverse_map]
                elif feat_out.shape[0] == N_original:
                    pt_feat = feat_out
                else:
                    inv = out.get("inverse") if isinstance(out, dict) else getattr(out, "inverse", None)
                    if inv is not None:
                        pt_feat = feat_out[inv]
                    else:
                        raise ValueError(f"提取失败！网络输出了 {feat_out.shape[0]} 个特征，但原始点有 {N_original} 个，且没有找到映射表！")

            assert pt_feat.shape[0] == N_original, f"严重错位！最终特征点数 {pt_feat.shape[0]} 与物理点数 {N_original} 不匹配！"

        torch.save({"point_features": pt_feat.cpu()}, out_path)

if __name__ == "__main__":
    main()