# process_image.py

import os
import cv2
import numpy as np
from crack_predict_code.predict import run_prediction
from crack_quantification.quantifier import compute_features

def process_image(image_path: str, pixel_size_mm: float = 0.1) -> dict:
    """
    加载图像，进行分割预测与量化分析。
    返回包含几何特征的字典。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ 文件不存在：{image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ 图像读取失败：{image_path}")

    print(f"🔍 正在处理图像：{image_path}，尺寸：{img.shape}")

    # 1. 分割预测（掩膜值为0或1）
    mask = run_prediction(img)

    # 2. 保存掩膜图像（用于调试）
    os.makedirs("output", exist_ok=True)
    mask_save_path = os.path.join("output", os.path.basename(image_path).replace(".jpg", "_mask.png"))
    cv2.imwrite(mask_save_path, (mask * 255).astype("uint8"))
    print(f"📤 掩膜已保存：{mask_save_path}")

    # 3. 几何量化分析（掩膜需转换为 uint8 图）
    result = compute_features((mask * 255).astype("uint8"), pixel_size_mm)
    return result
