import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_user_intent(user_input: str) -> dict:
    """
    Use GPT to analyze user input and return structured tool call:
    {
        "tool": "<tool_name>",
        "parameters": {
            "image_path": "...",
            "pixel_size": 0.2
        }
    }
    """
    prompt = f'''
You are an intelligent assistant for a crack analysis system.

Your job is to:
1. Identify which tool to call based on the user's request.
2. Extract the appropriate parameters and return a valid JSON object.

Available tools:
- analyze_all_images: analyze all images (accepts optional pixel_size in mm).
- segment_image: segment one image and save the mask (requires image_path).
- analyze_one_image: analyze a single image with quantification (requires image_path, optional pixel_size).
- summarize_results: summarize analysis results (no parameters).

Synonyms for segmentation:
- The following user expressions should always be interpreted as segment_image:
    - "visualize"
    - "display"
    - "segment"
    - "show"
    - "process"
    - "visualize the segmentation result"
    - "show prediction mask"
    - "see result image"

Synonyms for quantitative analysis:
- Expressions like "analyze", "quantify", "measure", "compute" should map to analyze_one_image.

Interpretations:
- "first image" → image_index: 0
- "second image" → image_index: 1
- "third image" → image_index: 2
- "fourth image" → image_index: 3
- "fifth image" → image_index: 4
- "last image" → image_index: -1

Important logic:
- If the user asks a follow-up question like "what's the max width", "what is the area", "is it compliant", or "how long is the crack",
  → and they did NOT mention anything about 'all images', 'summary', 'report', or 'result file',
  → then assume they are referring to the most recently analyzed single image.
  → In that case, return:
  {{
    "tool": "none",
    "parameters": {{ "query": "max width" }}
  }}

- If the user asks for repair suggestions, assessment, recommendation, or what to do next,
  and they do NOT mention 'all images' or 'summary',
  → assume they are referring to the last analyzed image.
  → In that case, return:
  {{
    "tool": "none",
    "parameters": {{ "query": "repair advice" }}
  }}

- If the user states the pixel size or scale, for example "pixel size is 0.2" or "scale 0.2 mm per pixel",
  → return:
  {{
    "tool": "none",
    "parameters": {{ "pixel_size": 0.2 }}
  }}

Examples:

User: "visualize the first image"
→ {{
  "tool": "segment_image",
  "parameters": {{
    "image_path": "1"
  }}
}}

User: "visualize input_images/8_crack.jpg"
→ {{
  "tool": "segment_image",
  "parameters": {{
    "image_path": "input_images/8_crack.jpg"
  }}
}}

User: "segment input_images/2_crack.jpg"
→ {{
  "tool": "segment_image",
  "parameters": {{
    "image_path": "input_images/2_crack.jpg"
  }}
}}

User: "analyze input_images/2_crack.jpg with pixel size 0.25mm"
→ {{
  "tool": "analyze_one_image",
  "parameters": {{
    "image_path": "input_images/2_crack.jpg",
    "pixel_size": 0.25
  }}
}}

User: "show all images at 0.1mm resolution"
→ {{
  "tool": "analyze_all_images",
  "parameters": {{
    "pixel_size": 0.1
  }}
}}

User: "summarize the result"
→ {{
  "tool": "summarize_results",
  "parameters": {{}}
}}

User: "pixel size is 0.1 mm per pixel"
→ {{
  "tool": "none",
  "parameters": {{ "pixel_size": 0.1 }}
}}

Important:
- Do NOT invent tool names.
- Only return valid JSON. No explanation.
- Tool names must be: segment_image, analyze_one_image, analyze_all_images, summarize_results, none

Now process the following user request and output JSON only:
"""{user_input}"""
'''

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        return {
            "tool": None,
            "parameters": {},
            "error": f"{type(e).__name__}: {e}"
        }


def resolve_image_path(params: dict) -> str:
    """
    Converts image_path or image_index into actual file path in input_images/.
    """
    folder = "input_images"
    if not os.path.exists(folder):
        raise FileNotFoundError("❌ input_images folder not found.")

    candidates = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not candidates:
        raise FileNotFoundError("❌ No images found in input_images.")

    if "image_index" in params:
        idx = params["image_index"]
        if idx < 0 or idx >= len(candidates):
            raise IndexError(f"Index {idx} out of range.")
        return os.path.join(folder, candidates[idx])

    path = params.get("image_path")
    if isinstance(path, str):
        if path.startswith("auto_first"):
            return os.path.join(folder, candidates[0])
        elif path.startswith("auto_last"):
            return os.path.join(folder, candidates[-1])
        else:
            return path

    raise ValueError("No image path/index provided.")