import os
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
import numpy as np

DATA_ROOT = os.environ.get("DATA_ROOT", "/root/autodl-tmp/nuscenes_mini")
DINO_DIR  = os.environ.get("DINO_DIR",  "/root/autodl-tmp/dino_features")
PTV3_DIR  = os.environ.get("PTV3_DIR",  "/root/autodl-tmp/ptv3_features")
SPLIT = os.environ.get("SPLIT", "mini_train")
OUT_SUBDIR = "train" if SPLIT == "mini_train" else "val"
READ_DINO_LAYER = os.environ.get("READ_DINO_LAYER", "layer_9")

DEFAULT_IMG_HW = (900, 1600)
SAVE_DIR = os.environ.get(
    "SAVE_DIR",
    f"/root/autodl-tmp/data_features/{READ_DINO_LAYER}/{OUT_SUBDIR}"
)
os.makedirs(SAVE_DIR, exist_ok=True)

splits = create_splits_scenes()
target_scenes = set(splits[SPLIT])

nusc = NuScenes(version="v1.0-mini", dataroot=DATA_ROOT, verbose=False)
for sample in nusc.sample:
    scene_name = nusc.get("scene", sample["scene_token"])["name"]
    if scene_name not in target_scenes:
        continue

    token = sample["token"]
    out_path = os.path.join(SAVE_DIR, f"{token}.pt")
    if os.path.exists(out_path):
        continue

    dino = torch.load(os.path.join(DINO_DIR, f"{token}.pt"))
    ptv3 = torch.load(os.path.join(PTV3_DIR, f"{token}.pt"))

    img_feat_dict = dino["image_features"]
    if READ_DINO_LAYER not in img_feat_dict:
        raise KeyError(f"{READ_DINO_LAYER} not found; available: {list(img_feat_dict.keys())}")
    img_feat = img_feat_dict[READ_DINO_LAYER]

    if img_feat.dim() == 3:
        B, L, C = img_feat.shape
        Hf = dino.get("feat_shape", None)
        if Hf is not None:
            Hf, Wf = int(Hf[0]), int(Hf[1])
        else:
            Hf, Wf = 46, 82
        img_feat = img_feat.view(B, Hf, Wf, C).contiguous()
    else:
        _, Hf, Wf, C = img_feat.shape

    if "img_shape" in dino:
        Himg, Wimg = int(dino["img_shape"][0]), int(dino["img_shape"][1])
    else:
        Himg, Wimg = DEFAULT_IMG_HW

    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pc = LidarPointCloud.from_file(os.path.join(DATA_ROOT, lidar_sd["filename"]))
    points = torch.tensor(pc.points.T, dtype=torch.float32)

    p_feat = ptv3["point_features"]
    if p_feat.shape[0] != points.shape[0]:
         raise ValueError(f"严重错误: Token {token} 的特征点数 {p_feat.shape[0]} 与物理点数 {points.shape[0]} 不匹配！这会导致标签错位。请重新提取 ptv3 特征。")

    map_dict = {
        1: 0, 5: 0, 7: 0, 8: 0, 10: 0, 11: 0, 13: 0, 19: 0, 20: 0, 0: 0, 
        29: 0, 31: 0, 9: 1, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 21: 6, 2: 7, 
        3: 7, 4: 7, 6: 7, 12: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 
        27: 14, 28: 15, 30: 16
    }
    learning_map = np.zeros(32, dtype=np.int64)
    for k, v in map_dict.items():
        learning_map[k] = v

    try:
        lidseg = nusc.get("lidarseg", lidar_sd["token"])
        raw_labels = np.fromfile(os.path.join(DATA_ROOT, lidseg["filename"]), dtype=np.uint8)
        labels = learning_map[raw_labels]
    except Exception:
        labels = np.zeros(points.shape[0], dtype=np.int64)

    torch.save(
        {
            "point_features": p_feat,                         # 保证对齐的特征
            "voxel_coords": points[:, :3],                    # xyz
            "labels": torch.tensor(labels, dtype=torch.long), 
            "image_features": img_feat.float(),               
            "calib_matrices": dino["calib_matrices"][:, :3],  
            "img_shape": torch.tensor([Himg, Wimg], dtype=torch.float32),
            "feat_shape": torch.tensor([Hf, Wf], dtype=torch.float32),
        },
        out_path,
    )