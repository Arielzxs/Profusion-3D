import os
import sys
import csv
import torch
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes

# Pointcept path
sys.path.append("/root/autodl-tmp/Pointcept")
from pointcept.models.builder import build_model as pc_build_model
from pointcept.utils.config import Config
from pointcept.datasets.transform import GridSample

DATA_ROOT = "/root/autodl-tmp/nuscenes_mini"
CKPT = "/root/autodl-tmp/ptv3_ckpt.pth"
CFG_PATH = "/root/autodl-tmp/Pointcept/configs/nuscenes/semseg-pt-v3m1-0-base.py"
OUT_CSV = "logs/official_ptv3_mini_val.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID_SIZE = 0.05


def build_learning_map():
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


def ensure_dict_output(data_out):
    if isinstance(data_out, dict):
        return data_out
    if isinstance(data_out, list):
        if len(data_out) == 0:
            return None
        if not isinstance(data_out[0], dict):
            raise TypeError(f"Unexpected GridSample list item type: {type(data_out[0])}")
        return data_out[0]
    raise TypeError(f"Unexpected GridSample output type: {type(data_out)}")


def main():
    os.makedirs("logs", exist_ok=True)

    print(f"[env] torch={torch.__version__}, device={DEVICE}")
    print(f"[path] data_root={DATA_ROOT}")
    print(f"[path] ckpt={CKPT}")
    print(f"[path] cfg={CFG_PATH}")

    assert os.path.exists(CKPT), f"checkpoint not found: {CKPT}"
    assert os.path.exists(CFG_PATH), f"config not found: {CFG_PATH}"
    assert os.path.exists(DATA_ROOT), f"data root not found: {DATA_ROOT}"

    cfg = Config.fromfile(CFG_PATH)
    model = pc_build_model(cfg.model)

    state = torch.load(CKPT, map_location="cpu")
    state = state.get("state_dict", state)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("[load] missing sample:", missing[:10])
    if unexpected:
        print("[load] unexpected sample:", unexpected[:10])

    model.to(DEVICE).eval()

    nusc = NuScenes(version="v1.0-mini", dataroot=DATA_ROOT, verbose=False)
    learning_map = build_learning_map()
    grid_transform = GridSample(grid_size=GRID_SIZE, hash_type="fnv", mode="test", return_inverse=True)
    grid_size_tensor = torch.tensor([GRID_SIZE, GRID_SIZE, GRID_SIZE], device=DEVICE, dtype=torch.float32)

    # 只评测 mini_val
    splits = create_splits_scenes()
    mini_val_scenes = set(splits["mini_val"])
    samples = [s for s in nusc.sample if nusc.get("scene", s["scene_token"])["name"] in mini_val_scenes]
    print(f"[split] mini_val samples: {len(samples)}")

    num_classes = 17
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i, sample in enumerate(tqdm(samples, desc="Official PTv3 eval (mini_val)")):
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        pc = LidarPointCloud.from_file(os.path.join(DATA_ROOT, lidar_sd["filename"]))

        points = pc.points[:3].T.astype(np.float32)      # [N,3]
        intensity = pc.points[3:4].T.astype(np.float32)  # [N,1]

        lidseg = nusc.get("lidarseg", lidar_sd["token"])
        raw_labels = np.fromfile(os.path.join(DATA_ROOT, lidseg["filename"]), dtype=np.uint8)
        labels = learning_map[raw_labels]  # [N], 0~16

        data_in = {
            "coord": points,
            "segment": labels,
            "strength": intensity
        }
        data_out = grid_transform(data_in)
        data_dict = ensure_dict_output(data_out)
        if data_dict is None:
            continue

        coord_np = np.asarray(data_dict["coord"], dtype=np.float32)
        strength_np = np.asarray(data_dict["strength"], dtype=np.float32)
        inverse_np = np.asarray(data_dict["inverse"], dtype=np.int64) if "inverse" in data_dict else None

        coord = torch.tensor(coord_np, dtype=torch.float32, device=DEVICE)
        strength = torch.tensor(strength_np, dtype=torch.float32, device=DEVICE)
        feat = torch.cat([coord, strength], dim=1)  # [M,4]
        grid_coord = torch.floor(coord / grid_size_tensor).to(torch.int32)

        input_dict = {
            "coord": coord,
            "grid_coord": grid_coord,
            "offset": torch.tensor([coord.shape[0]], dtype=torch.int64, device=DEVICE),
            "feat": feat,
            "batch": torch.zeros(coord.shape[0], dtype=torch.long, device=DEVICE),
        }
        if inverse_np is not None:
            input_dict["inverse"] = torch.tensor(inverse_np, dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            out = model(input_dict)
            if isinstance(out, dict):
                out = out.get("seg_logits", out.get("logits", out.get("feat", out)))
            if not torch.is_tensor(out):
                raise TypeError(f"Model output type error: {type(out)}")

            # 回填到原始点
            if out.shape[0] != len(labels):
                if "inverse" not in input_dict:
                    raise RuntimeError(
                        f"Need inverse for point recovery: out={out.shape[0]}, raw={len(labels)}"
                    )
                out = out[input_dict["inverse"]]

            pred = out.argmax(1).cpu().numpy()
            gt = labels

            if i < 3:
                print(
                    f"[debug {i}] rawN={len(gt)}, outN={len(pred)}, "
                    f"pred_unique={np.unique(pred)[:10]}, gt_unique={np.unique(gt)[:10]}"
                )

            mask = gt != 0  # ignore class
            for g, p in zip(gt[mask], pred[mask]):
                if 0 <= g < num_classes and 0 <= p < num_classes:
                    hist[g, p] += 1

    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-6)
    miou = float(np.nanmean(iu[1:]))
    acc = np.diag(hist) / (hist.sum(1) + 1e-6)
    macc = float(np.nanmean(acc[1:]))
    freq = hist.sum(1) / (hist.sum() + 1e-6)
    fwiou = float(np.nansum(freq * iu))

    print(f"\n[Official PTv3 mini_val] mIoU={miou:.4f}, mAcc={macc:.4f}, fwIoU={fwiou:.4f}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "miou", "macc", "fwiou", "lr", "model"])
        w.writerow([1, 0.0, miou, macc, fwiou, 0.0, "official_ptv3_mini"])

    print(f"[saved] {OUT_CSV}")


if __name__ == "__main__":
    main()