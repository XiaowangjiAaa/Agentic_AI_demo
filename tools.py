import os
import re
import cv2
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from agent_main import main as process_all_images, process_image
from crack_predict_code.predict import run_prediction


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========= Tool 1: analyze_all_images =========
def analyze_all_images(pixel_size: float = 0.1) -> str:
    if os.path.exists("output/result_metrics.csv"):
        return f"✅ Results already exist (pixel size {pixel_size} mm): output/result_metrics.csv, no need to re-analyze."
    process_all_images(pixel_size_mm=pixel_size)
    return f"✅ All images have been processed (pixel size {pixel_size} mm). Results saved to output/result_metrics.csv"

analyze_all_images_spec = {
    "name": "analyze_all_images",
    "description": "Re-analyze all images in the input_images folder: perform segmentation and quantification",
    "parameters": {
        "type": "object",
        "properties": {
            "pixel_size": {
                "type": "number",
                "description": "Pixel size in millimeters",
                "default": 0.1
            }
        },
        "required": []
    }
}

# ========= Tool 2: segment_image =========
def segment_image(image_path: str) -> str:
    if not os.path.exists(image_path):
        return f"❌ File not found: {image_path}"
    try:
        img = cv2.imread(image_path)
        if img is None:
            return f"❌ Image read failed: {image_path}"
        mask = run_prediction(img)
        os.makedirs("output/result_images", exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join("output/result_images", f"{base}_mask.png")
        cv2.imwrite(mask_path, (mask * 255).astype("uint8"))
        return f"✅ Segmentation completed for {os.path.basename(image_path)}. Mask saved to {mask_path}"
    except Exception as e:
        return f"❌ Segmentation failed: {type(e).__name__} - {e}"

segment_image_spec = {
    "name": "segment_image",
    "description": "Segment a crack image and save the mask without quantification",
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {"type": "string", "description": "Image path"}
        },
        "required": ["image_path"]
    }
}

# ========= Tool 3: analyze_one_image =========
def analyze_one_image(image_path: str, pixel_size: float = 0.1) -> str:
    if not os.path.exists(image_path):
        return f"❌ File not found: {image_path}"
    try:
        result = process_image(image_path, pixel_size_mm=pixel_size)
        return f"✅ Successfully analyzed: {os.path.basename(image_path)} (pixel size {pixel_size} mm)\n" + \
               "\n".join([f"{k}: {v}" for k, v in result.items()])
    except Exception as e:
        return f"❌ Analysis failed: {type(e).__name__} - {e}"

analyze_one_image_spec = {
    "name": "analyze_one_image",
    "description": "Analyze a specific image, e.g., input_images/7_crack.jpg, and extract crack features",
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {"type": "string", "description": "Image path, e.g., input_images/7_crack.jpg"},
            "pixel_size": {"type": "number", "description": "Pixel size in millimeters", "default": 0.1}
        },
        "required": ["image_path"]
    }
}

# ========= Tool 4: summarize_results =========
def summarize_results() -> str:
    csv_path = "output/result_metrics.csv"
    if not os.path.exists(csv_path):
        return "❌ Result file not found. Please run image analysis first."

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return "❌ Result file is found but empty. Please check if analysis completed successfully."
        if "Compliance" not in df.columns:
            compliance_columns = ["Max Width OK", "Avg Width OK", "Area Ratio OK", "Length OK"]
            if all(col in df.columns for col in compliance_columns):
                df["Compliance"] = df[compliance_columns].all(axis=1)
            else:
                return "❌ Cannot summarize: missing required compliance fields in result file."

        prompt = f'''
You are a crack analysis expert. The following is the quantified crack analysis result:
{df.to_string(index=False)}

Please summarize in natural language:
1. How many images were analyzed?
2. Which image has the maximum crack width?
3. How many images are non-compliant (due to max width, area, or length)?
4. Provide a general assessment and suggestions.
'''
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ GPT summarization failed: {type(e).__name__}: {e}"

summarize_results_spec = {
    "name": "summarize_results",
    "description": "Read output/result_metrics.csv and generate a crack analysis summary including max width, compliance issues, and suggestions",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

# ========= Tool 5: fetch_web_content =========
def fetch_web_content(url: str) -> str:
    """Fetch plain text content from a web page."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return f"❌ Failed to fetch {url}: {e}"

    text = response.text
    if len(text) > 3000:
        text = text[:3000]
    return text

fetch_web_content_spec = {
    "name": "fetch_web_content",
    "description": "Fetch raw text content from a web page for reference or summarization.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Target URL"}
        },
        "required": ["url"]
    }
}

# ========= Path Extraction (UI support) =========
def extract_image_paths(text: str) -> dict:
    match = re.search(r"(input_images[\\/][\w\-.]+)", text)
    if not match:
        return {}
    base = os.path.splitext(os.path.basename(match.group(1)))[0]
    return {
        "original": f"input_images/{base}.jpg",
        "mask": f"output/result_images/{base}_mask.png",
        "width": f"output/result_images/{base}_width.png",
    }

# ========= Register tools for Agent usage =========
FUNCTION_SCHEMAS = [
    analyze_all_images_spec,
    segment_image_spec,
    analyze_one_image_spec,
    summarize_results_spec,
    fetch_web_content_spec,
]

FUNCTION_MAP = {
    "analyze_all_images": lambda args: analyze_all_images(**args),
    "segment_image": lambda args: segment_image(**args),
    "analyze_one_image": lambda args: analyze_one_image(**args),
    "summarize_results": lambda args: summarize_results(),
    "fetch_web_content": lambda args: fetch_web_content(**args),
}
