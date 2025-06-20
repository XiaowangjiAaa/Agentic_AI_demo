# crack_predict_code/predict.py

import torch
import numpy as np
from torchvision import transforms
from PIL import Image as PILImage
from crack_detection_model.unet import UNet
import cv2

# 初始化模型（只加载一次）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=3, num_classes=1)
model.load_state_dict(torch.load(r"crack_predict_code\unet_best.pth", map_location=device))  # <-- 确保路径正确
model.to(device)
model.eval()

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((896, 896)),
    transforms.ToTensor()
])

def run_prediction(image_np: np.ndarray) -> np.ndarray:
    """
    输入: OpenCV 读取的 RGB 图像 (np.ndarray, HWC)
    输出: 分割掩膜 (np.ndarray, HW)，值为0或1
    """
    pil_img = PILImage.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.sigmoid(pred)
        pred_mask = (pred > 0.5).float().squeeze().cpu().numpy()  # shape: (H, W)

    return pred_mask  # 值为 0. 或 1.
