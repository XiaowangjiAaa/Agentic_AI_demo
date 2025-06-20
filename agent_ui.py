import gradio as gr
from agent_executor import agent_respond

# UI 响应逻辑：支持记忆式 Agent 返回

def run_interface(user_input):
    response, paths = agent_respond(user_input)
    return (
        paths.get("original"), 
        paths.get("mask"), 
        paths.get("width"), 
        None,  # 第四张图保留
        response
    )

with gr.Blocks(
    theme="soft",
    css="""
#agent_output .wrap.svelte-1ipelgc {
    font-size: 18px !important;
    line-height: 1.8;
    font-family: 'Microsoft YaHei', sans-serif;
    white-space: pre-wrap;
}
"""
) as demo:
    gr.Markdown("## 🧠 Agentic AI - UTS AI4C Lab")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                img1 = gr.Image(label="Input image", height=400)
                img2 = gr.Image(label="Predicted", height=400)
            with gr.Row():
                img3 = gr.Image(label="Maximum Crack width", height=400)
                img4 = gr.Image(label="Haaaa!!!", height=400)

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Agent AI", 
                lines=30, 
                interactive=False, 
                elem_id="agent_output"
            )
            user_input = gr.Textbox(
                label="Please input tasks!!!", 
                placeholder="Example: Visualize input_images/8_crack.jpg"
            )
            submit = gr.Button("Submit")

    # 提交与 Enter 键触发一致
    submit.click(
        fn=run_interface, 
        inputs=user_input, 
        outputs=[img1, img2, img3, img4, output_text]
    )
    user_input.submit(
        fn=run_interface, 
        inputs=user_input, 
        outputs=[img1, img2, img3, img4, output_text]
    )

if __name__ == "__main__":
    demo.launch(share=True)
