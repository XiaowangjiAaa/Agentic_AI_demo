import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from tools import (
    FUNCTION_MAP as function_map,
    FUNCTION_SCHEMAS as functions,
    extract_image_paths
)
from agent_parser import (
    parse_user_intent,
    resolve_auto_image_path
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========== CLI Entry ========== #
def run_agent():
    print("🤖 Crack Analysis Agent is ready. Enter your instruction ('exit' to quit):")
    history = []
    last_image_result = None
    last_image_path = None

    while True:
        user_input = input("\n🗣️ You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # 如果用户说“请解释这些结果”、“这些参数说明什么”，自动引用上次结果
        if last_image_result and any(x in user_input.lower() for x in ["explain", "analyze this", "these results", "these parameters", "what do they mean"]):
            print("📎 Using memory of last result...")
            messages = history + [
                {"role": "user", "content": "Please analyze the following crack measurement result:"},
                {"role": "function", "name": "analyze_one_image", "content": last_image_result},
                {"role": "user", "content": user_input}
            ]
            followup = client.chat.completions.create(model="gpt-4o", messages=messages)
            reply = followup.choices[0].message.content.strip()
            print(f"\n🤖 {reply}")
            history += messages[-2:] + [{"role": "assistant", "content": reply}]
            continue

        # 正常请求流程
        messages = history + [{"role": "user", "content": user_input}]
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                functions=functions,
                function_call="auto"
            )
            msg = response.choices[0].message

            if msg.function_call:
                fn_name = msg.function_call.name
                args = json.loads(msg.function_call.arguments)

                # 自动路径替换
                if "image_path" in args and args["image_path"].startswith("auto_"):
                    args["image_path"] = resolve_auto_image_path(args["image_path"])

                fn = function_map.get(fn_name)
                if not fn:
                    reply = f"❌ Unknown function: {fn_name}"
                else:
                    print(f"🔧 Running tool: {fn_name}({args})")
                    result = fn(args)

                    # 如果是图像分析，记住结果
                    if fn_name == "analyze_one_image" and "image_path" in args:
                        last_image_path = args["image_path"]
                        last_image_result = result

                    messages.append(msg)
                    messages.append({"role": "function", "name": fn_name, "content": result})

                    followup = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages
                    )
                    reply = followup.choices[0].message.content
            else:
                reply = msg.content

            print(f"\n🤖 {reply}")
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})

        except Exception as e:
            print(f"❌ Error: {type(e).__name__} - {e}")



# ========== For UI (Gradio/Streamlit etc.) ========== #
def agent_respond(user_input: str):
    """
    Used in visual UI to return both reply and image path dict.
    """
    messages = [{"role": "user", "content": user_input}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        functions=functions,
        function_call="auto"
    )

    msg = response.choices[0].message
    reply = ""

    if msg.function_call:
        fn_name = msg.function_call.name
        args = json.loads(msg.function_call.arguments)
        if "image_path" in args and args["image_path"].startswith("auto_"):
            args["image_path"] = resolve_auto_image_path(args["image_path"])
        fn = function_map[fn_name]
        result = fn(args)
        reply = result
    else:
        reply = msg.content

    # Optional: extract image paths for display
    img_paths = extract_image_paths(user_input)
    return reply, img_paths


# ========== Optional (Deprecated): parse_user_intent fallback ========== #
def handle_user_request(user_input: str) -> str:
    parsed = parse_user_intent(user_input)
    tool = parsed.get("tool")
    params = parsed.get("parameters", {})

    if tool not in function_map:
        return f"❌ Could not determine the correct tool.\nDetails: {parsed.get('error', '')}"

    if "image_path" in params and params["image_path"].startswith("auto_"):
        try:
            params["image_path"] = resolve_auto_image_path(params["image_path"])
        except Exception as e:
            return f"❌ Failed to resolve image path: {e}"

    return function_map[tool](params)


# ========== Main CLI Entry ========== #
if __name__ == "__main__":
    run_agent()
