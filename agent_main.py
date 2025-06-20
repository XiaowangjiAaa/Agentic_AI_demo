# agent_main.py

import os
import cv2
import numpy as np
import pandas as pd
from crack_predict_code.predict import run_prediction
from crack_quantification.quantifier import compute_features


def process_image(image_path, pixel_size_mm=0.1):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")

    # 1. 模型预测
    mask = run_prediction(image)  # float32, 0./1.
    mask_uint8 = (mask * 255).astype(np.uint8)

    # 2. 特征提取应使用 uint8 掩膜
    features = compute_features(mask_uint8, pixel_size_mm=pixel_size_mm)

    # 3. 保存图像
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = "output/result_images"
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.png"), mask_uint8)

    width_vis = features.get("width_visualization")
    if width_vis is not None:
        if width_vis.dtype != np.uint8:
            width_vis = (width_vis * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_width.png"), width_vis)

    # 4. 返回结构化数据
    result = {
        "Filename": os.path.basename(image_path),
        "Max Width (mm)": features["Max Width (mm)"],
        "Avg Width (mm)": features["Avg Width (mm)"],
        "Length (mm)": features["Length (mm)"],
        "Area (mm^2)": features["Area (mm^2)"],
        "Area Ratio": features["Area Ratio (%)"],
        "Max Width OK": features["Compliance"]["Max Width OK"],
        "Avg Width OK": features["Compliance"]["Avg Width OK"],
        "Area Ratio OK": features["Compliance"]["Area Ratio OK"],
        "Length OK": features["Compliance"]["Length OK"],
    }

    return result



def main(pixel_size_mm=0.1):
    """批量处理 input_images 中所有图像并保存 CSV"""
    input_dir = "input_images"
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results = []

    for fname in image_files:
        try:
            path = os.path.join(input_dir, fname)
            print(f"分析图像：{fname}")
            result = process_image(path, pixel_size_mm=pixel_size_mm)
            results.append(result)
        except Exception as e:
            print(f"❌ 处理 {fname} 失败：{e}")

    # 保存所有结果
    if results:
        df = pd.DataFrame(results)
        os.makedirs("output", exist_ok=True)
        df.to_csv("output/result_metrics.csv", index=False)
        print("✅ 所有结果已保存到 output/result_metrics.csv")
