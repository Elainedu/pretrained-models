import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 載入模型和分詞器（從當前目錄）
model_path = "."  # 當前目錄包含 model.safetensors 和 config.json
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 定義生成回應的函數
def generate_response(prompt, max_new_tokens=128, temperature=0.7, top_p=0.9, top_k=50):
    """
    生成對話回應
    Args:
        prompt: 使用者輸入
        max_new_tokens: 最大生成 token 數量
        temperature: 溫度參數（控制隨機性）
        top_p: nucleus sampling 參數
        top_k: top-k sampling 參數
    """
    # 將輸入編碼
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 生成回應
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 解碼輸出（只返回新生成的部分）
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# 創建 Gradio 接口
with gr.Blocks(title="BLOOMZ-396M 中文對話模型") as demo:
    gr.Markdown("# 🤖 BLOOMZ-396M 中文對話模型")
    gr.Markdown("基於 YeungNLP/bloomz-396m-zh 的中文對話生成模型")

    with gr.Row():
        with gr.Column(scale=3):
            user_input = gr.Textbox(
                label="輸入您的問題",
                placeholder="請輸入您想問的問題...",
                lines=3
            )
            submit_btn = gr.Button("生成回應", variant="primary")

        with gr.Column(scale=1):
            max_tokens = gr.Slider(
                minimum=32,
                maximum=512,
                value=128,
                step=32,
                label="最大生成長度"
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.7,
                step=0.1,
                label="Temperature（創造性）"
            )
            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top-p"
            )
            top_k = gr.Slider(
                minimum=10,
                maximum=100,
                value=50,
                step=10,
                label="Top-k"
            )

    output = gr.Textbox(
        label="模型回應",
        lines=8,
        interactive=False
    )

    # 範例問題
    gr.Examples(
        examples=[
            "你好，請自我介紹一下",
            "什麼是深度學習？",
            "如何學習Python程式設計？",
            "請推薦一本好書"
        ],
        inputs=user_input
    )

    # 綁定事件
    submit_btn.click(
        fn=generate_response,
        inputs=[user_input, max_tokens, temperature, top_p, top_k],
        outputs=output
    )

    user_input.submit(
        fn=generate_response,
        inputs=[user_input, max_tokens, temperature, top_p, top_k],
        outputs=output
    )

# 啟動 Gradio 介面
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
