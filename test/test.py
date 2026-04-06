import torch
import sys
sys.path.insert(0, "/root/dinov2")  # 确保能找到 dinov2 包

# 使用 hubconf 加载（与脚本一致）
model = torch.hub.load("/root/dinov2", "dinov2_vits14", source="local", pretrained=False)
print("Model structure loaded")

# 加载离线权重（假设文件存在）
state = torch.load("/root/offline/dinov2_vits14_pretrain.pth", map_location="cpu")
model.load_state_dict(state)
print("Weights loaded")

model = model.cuda().eval()
print("Model moved to GPU")