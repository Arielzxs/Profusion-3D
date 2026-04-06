import os, sys, torch, numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import argparse

# 默认路径，可按需修改
DINO_REPO   = "/root/dinov2"
STATE_PATH  = "/root/offline/dinov2_vits14_pretrain.pth"
DATA_ROOT   = "/root/autodl-tmp/nuscenes_mini"
SAVE_DIR    = "/root/autodl-tmp/dino_features"

transform = T.Compose([
    T.Resize((644, 1148)),
    T.ToTensor(),
    T.Normalize(mean=[123.675/255., 116.28/255., 103.53/255.],
                std=[58.395/255., 57.12/255., 57.375/255.])
])
# 计算从 LiDAR 坐标系到指定相机图像像素坐标系的 4×4 投影矩阵
def lidar2img(nusc, sample, cam_channel):
    cam = nusc.get('sample_data', sample['data'][cam_channel])
    cam_calib = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    cam_pose  = nusc.get('ego_pose', cam['ego_pose_token'])

    lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_calib = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
    lidar_pose  = nusc.get('ego_pose', lidar['ego_pose_token'])

    lidar2ego = np.eye(4); lidar2ego[:3,:3] = Quaternion(lidar_calib['rotation']).rotation_matrix
    lidar2ego[:3, 3] = lidar_calib['translation']

    ego2global = np.eye(4); ego2global[:3,:3] = Quaternion(lidar_pose['rotation']).rotation_matrix
    ego2global[:3, 3] = lidar_pose['translation']

    global2camego = np.eye(4); global2camego[:3,:3] = Quaternion(cam_pose['rotation']).rotation_matrix.T
    global2camego[:3, 3] = -global2camego[:3,:3] @ np.array(cam_pose['translation'])

    camego2cam = np.eye(4); camego2cam[:3,:3] = Quaternion(cam_calib['rotation']).rotation_matrix.T
    camego2cam[:3, 3] = -camego2cam[:3,:3] @ np.array(cam_calib['translation'])

    K = np.eye(4); K[:3,:3] = np.array(cam_calib['camera_intrinsic'])

    return K @ camego2cam @ global2camego @ ego2global @ lidar2ego

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=DATA_ROOT)
    p.add_argument("--save_dir",  default=SAVE_DIR)
    p.add_argument("--state_path", default=STATE_PATH)
    p.add_argument("--dino_repo",  default=DINO_REPO)
    # 1-based 层号，逗号分隔，如 "9" 或 "9,12"
    p.add_argument("--layers", type=str, default=os.getenv("DINO_LAYERS", "9"))
    p.add_argument("--version", default="v1.0-mini")
    return p.parse_args()

def main():
    args = parse_args()
    # 把 1-based 层号转换为 0-based 索引，并校验
    raw_layers = [int(x) for x in args.layers.split(",")]
    layer_idx  = []
    for lid in raw_layers:
        idx = lid - 1            # 1-based -> 0-based
        if idx < 0:
            raise ValueError(f"Layer {lid} 无效")
        layer_idx.append(idx)

    os.makedirs(args.save_dir, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    cams = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DINOv2 on {device}, layers={raw_layers} (0-based {layer_idx})")

    dino = torch.hub.load(args.dino_repo, 'dinov2_vits14', source='local', pretrained=False)
    state = torch.load(args.state_path, map_location=device)
    dino.load_state_dict(state)
    dino = dino.to(device).eval()

    num_layers = len(dino.blocks)
    for idx in layer_idx:
        if idx >= num_layers:
            raise ValueError(f"请求层 {idx} 超出模型层数 {num_layers}")

    for sample in tqdm(nusc.sample, desc="DINO feat"):
        token = sample["token"]
        out_path = os.path.join(args.save_dir, f"{token}.pt")
        if os.path.exists(out_path):
            continue

        imgs, mats = [], []
        for cam in cams:
            sd = nusc.get('sample_data', sample['data'][cam])
            img_path = os.path.join(args.data_root, sd['filename'])
            img = Image.open(img_path).convert("RGB")
            imgs.append(transform(img))
            mats.append(lidar2img(nusc, sample, cam))

        imgs = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats_list = dino.get_intermediate_layers(imgs, n=layer_idx, return_class_token=False)
        feats = {f"layer_{lid}": f.cpu() for lid, f in zip(raw_layers, feats_list)}

        torch.save({
            "image_features": feats,              # dict: layer_9 / layer_12 -> [6,3772,384]
            "calib_matrices": torch.tensor(np.array(mats), dtype=torch.float32)
        }, out_path)

if __name__ == "__main__":
    main()