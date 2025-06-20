import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from tools import (
    FUNCTION_MAP as function_map,
    FUNCTION_SCHEMAS as functions,
    extract_image_paths
)
from memory import Memory
from agent_parser import (
    parse_user_intent,
    resolve_image_path
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========== Persistent Memory ==========
memory = Memory()

def _result_to_dict(text: str):
    if not text:
        return None
    try:
        return {
            line.split(":")[0].strip(): line.split(":")[1].strip()
            for line in text.splitlines()[2:] if ":" in line
        }
    except Exception:
        return None

# ========== Utility: fallback path correction ==========
def try_correct_image_filename(wrong_path: str) -> str:
    folder = "input_images"
    if not os.path.exists(folder):
        return None
    base_name = os.path.splitext(os.path.basename(wrong_path))[0].lower()
    candidates = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for f in candidates:
        if base_name in f.lower():
            return os.path.join(folder, f)
    return None

# ========== CLI Entry ==========
def run_agent():
    print("🤖 Crack Analysis Agent is ready. Enter your instruction ('exit' to quit):")
    history = memory.get_history()
    last_result_text, last_image_path = memory.get_last_result()
    last_result_dict = _result_to_dict(last_result_text)

    while True:
        user_input = input("\n🗣️ You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        parsed = parse_user_intent(user_input)
        tool = parsed.get("tool")
        params = parsed.get("parameters", {})

        if not tool or tool not in function_map:
            query = user_input.lower()
            if last_result_dict:
                if "max width" in query:
                    print(f"\n🤖 Max Width: {last_result_dict.get('Max Width (mm)', 'N/A')} mm")
                    continue
                if "avg width" in query or "average width" in query:
                    print(f"\n🤖 Avg Width: {last_result_dict.get('Avg Width (mm)', 'N/A')} mm")
                    continue
                if "area" in query:
                    print(f"\n🤖 Area: {last_result_dict.get('Area (mm^2)', 'N/A')} mm²")
                    continue
                if "length" in query:
                    print(f"\n🤖 Length: {last_result_dict.get('Length (mm)', 'N/A')} mm")
                    continue
                if "compliant" in query:
                    flags = [k for k in last_result_dict if k.endswith("OK")]
                    status = {k: last_result_dict[k] for k in flags}
                    print(f"\n🤖 Compliance: {status}")
                    continue
                if any(k in query for k in ["advice", "repair", "fix", "how should", "what should"]):
                    summary_prompt = (
                        "You are a crack repair expert. Based on the following result, give suggestions:\n\n"
                        + json.dumps(last_result_dict, indent=2)
                    )
                    completion = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": summary_prompt}],
                        temperature=0.5,
                    )
                    reply = completion.choices[0].message.content.strip()
                    print(f"\n🤖 {reply}")
                    history.append({"role": "user", "content": user_input})
                    history.append({"role": "assistant", "content": reply})
                    memory.add_message("user", user_input)
                    memory.add_message("assistant", reply)
                    continue
            try:
                completion = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an intelligent assistant for a crack analysis system."},
                        {"role": "user", "content": user_input},
                    ],
                    temperature=0.3,
                )
                reply = completion.choices[0].message.content.strip()
                print(f"\n🤖 {reply}")
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": reply})
                memory.add_message("user", user_input)
                memory.add_message("assistant", reply)
            except Exception as e:
                print(f"❌ GPT fallback failed: {type(e).__name__}: {e}")
            continue

        try:
            if "image_index" in params or "image_path" in params:
                params["image_path"] = resolve_image_path(params)
        except Exception as e:
            print(f"❌ Failed to resolve image: {e}")
            continue

        if "image_path" in params and not os.path.exists(params["image_path"]):
            corrected = try_correct_image_filename(params["image_path"])
            if corrected:
                print(f"📎 Auto-corrected image path: {params['image_path']} → {corrected}")
                params["image_path"] = corrected

        params.pop("image_index", None)
        fn = function_map[tool]
        print(f"🔧 Running tool: {tool}({params})")
        result = fn(params)

        print(f"\n🤖 {result}")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": result})
        memory.add_message("user", user_input)
        memory.add_message("assistant", result)

        if tool == "analyze_one_image" and isinstance(result, str):
            last_result_text = result
            last_image_path = params.get("image_path")
            last_result_dict = _result_to_dict(result)
            memory.set_last_result(last_result_text, last_image_path)

# ========== For UI (Gradio etc.) ==========
def agent_respond(user_input: str):
    last_result_text, last_image_path = memory.get_last_result()
    last_result_dict = _result_to_dict(last_result_text)
    parsed = parse_user_intent(user_input)
    tool = parsed.get("tool")
    params = parsed.get("parameters", {})

    if not tool or tool not in function_map:
        query = user_input.lower()
        if last_result_dict:
            if "max width" in query:
                return f"Max Width: {last_result_dict.get('Max Width (mm)', 'N/A')} mm", {}
            if "area" in query:
                return f"Area: {last_result_dict.get('Area (mm^2)', 'N/A')} mm²", {}
            if "length" in query:
                return f"Length: {last_result_dict.get('Length (mm)', 'N/A')} mm", {}
            if any(k in query for k in ["advice", "repair", "fix", "how should", "what should"]):
                summary_prompt = f"You are a crack repair expert. Based on the following result, give suggestions:\n\n{json.dumps(last_result_dict, indent=2)}"
                completion = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.5
                )
                reply = completion.choices[0].message.content.strip()
                memory.add_message("user", user_input)
                memory.add_message("assistant", reply)
                return reply, {}

        try:
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an intelligent assistant for a crack analysis system."},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.3
            )
            reply = completion.choices[0].message.content.strip()
            memory.add_message("user", user_input)
            memory.add_message("assistant", reply)
            return reply, {}
        except Exception as e:
            return f"❌ GPT fallback failed: {type(e).__name__}: {e}", {}

    try:
        if "image_index" in params or "image_path" in params:
            params["image_path"] = resolve_image_path(params)
    except Exception as e:
        return f"❌ Failed to resolve image: {e}", {}

    if "image_path" in params and not os.path.exists(params["image_path"]):
        corrected = try_correct_image_filename(params["image_path"])
        if corrected:
            params["image_path"] = corrected

    params.pop("image_index", None)
    fn = function_map[tool]
    result = fn(params)

    if tool == "analyze_one_image" and isinstance(result, str):
        last_result_text = result
        last_image_path = params.get("image_path")
        last_result_dict = _result_to_dict(result)
        memory.set_last_result(last_result_text, last_image_path)

    image_reference = params.get("image_path", user_input)
    paths = extract_image_paths(image_reference)
    return result, paths

# ========== Optional Fallback ==========
def handle_user_request(user_input: str) -> str:
    parsed = parse_user_intent(user_input)
    tool = parsed.get("tool")
    params = parsed.get("parameters", {})

    if tool not in function_map:
        return f"❌ Could not determine the correct tool.\nDetails: {parsed.get('error', '')}"

    try:
        if "image_index" in params or "image_path" in params:
            params["image_path"] = resolve_image_path(params)
    except Exception as e:
        return f"❌ Failed to resolve image: {e}"

    if "image_path" in params and not os.path.exists(params["image_path"]):
        corrected = try_correct_image_filename(params["image_path"])
        if corrected:
            params["image_path"] = corrected

    params.pop("image_index", None)
    result = function_map[tool](params)
    memory.add_message("user", user_input)
    memory.add_message("assistant", result)
    if tool == "analyze_one_image" and isinstance(result, str):
        memory.set_last_result(result, params.get("image_path"))
    return result

# ========== Main Entry ==========
if __name__ == "__main__":
    run_agent()
