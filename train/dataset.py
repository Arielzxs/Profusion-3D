import os, torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        img_feats = data["image_features"].permute(0, 3, 1, 2).contiguous()  # [6,C,Hf,Wf]
        return {
            "point_features": data["point_features"],   # [V, C_lidar]
            "image_features": img_feats,                # [6,C,Hf,Wf]
            "voxel_coords":   data["voxel_coords"],     # [V,3]
            "labels":         data["labels"],           # [V]
            "calib_matrices": data["calib_matrices"],   # [6,3,4]
            "img_shape":      data["img_shape"],        # [2] H,W
            "feat_shape":     data["feat_shape"],       # [2] Hf,Wf
        }

def collate_list(batch):
    return batch  # 逐样本 list