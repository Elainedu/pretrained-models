"""
===============================================================================
專案名稱: BLOOMZ-396M Chinese Dialogue Demo - 中文對話生成展示
===============================================================================

[專案簡介]
這是一個使用預訓練 BLOOMZ-396M-ZH 模型的中文對話生成系統展示。
直接使用 Hugging Face 的預訓練模型，無需額外訓練，展示基礎語言模型
的對話生成能力，作為後續微調專案的對照基準。

[技術架構]
- 模型: BLOOMZ-396M-ZH (YeungNLP)
- 參數量: 396M (3.96億參數)
- 模型類型: Causal Language Model (自回歸語言模型)
- 深度學習框架: PyTorch + Transformers
- Web 介面: Gradio

[BLOOM 模型系列]
BLOOM (BigScience Large Open-science Open-access Multilingual Language Model)
- 開發者: BigScience 多國合作專案
- 特色: 開源多語言大型語言模型
- 訓練資料: 46 種自然語言和 13 種程式語言
- BLOOMZ: 在 BLOOM 基礎上進行 instruction tuning
- 中文版: YeungNLP/bloomz-396m-zh 針對中文優化

[模型規格]
- 參數量: 396M
- 層數: 30 layers
- 注意力頭: 16 heads
- 隱藏層維度: 1024
- 詞彙表大小: 250,880
- 精度: float16 (GPU) / float32 (CPU)

[生成參數說明]
1. Max New Tokens (32-512):
   - 控制生成文字的最大長度
   - 預設: 128 tokens

2. Temperature (0.1-1.5):
   - 控制生成的隨機性和創造性
   - 低溫 (0.1-0.5): 更確定、保守、重複性高
   - 中溫 (0.6-0.9): 平衡的創造性
   - 高溫 (1.0-1.5): 更多樣、創意、可能不連貫
   - 預設: 0.7

3. Top-p (0.1-1.0):
   - Nucleus Sampling（核採樣）
   - 從累積機率達到 p 的候選詞中採樣
   - 預設: 0.9

4. Top-k (10-100):
   - 只從機率最高的 k 個詞中選擇
   - 預設: 50

[啟動方式]
基本啟動:
    python gradio.py

指定 Port:
    python gradio.py  # 預設 7860

[使用說明]
1. 啟動後開啟 http://127.0.0.1:7860
2. 在輸入框輸入問題或指令
3. 點擊「生成回應」按鈕
4. 模型會基於輸入生成回應
5. 可嘗試範例問題或調整右側參數

[範例問題]
1. "你好，請自我介紹一下"
2. "什麼是深度學習？"
3. "如何學習Python程式設計？"
4. "請推薦一本好書"

[與微調模型的比較]
這個展示使用預訓練模型（未經任務特定微調）:
- 優點: 無需訓練、快速部署、通用性強
- 缺點: 可能不夠專注、回答較通用、領域知識有限

微調後的模型（如 QLoRA、全微調）:
- 優點: 針對特定任務優化、回答更精準、風格可控
- 缺點: 需要訓練資料和計算資源

[模型載入方式]
本專案從當前目錄載入模型檔案:
- model.safetensors: 模型權重
- config.json: 模型配置
- tokenizer files: 分詞器檔案

如需從 Hugging Face 下載:
```python
tokenizer = AutoTokenizer.from_pretrained('YeungNLP/bloomz-396m-zh')
model = AutoModelForCausalLM.from_pretrained('YeungNLP/bloomz-396m-zh')
```

[面試展示重點]
1. **預訓練基礎**: 說明大型語言模型的預訓練原理
2. **生成機制**: 解釋自回歸生成和採樣策略
3. **參數影響**: 展示不同 temperature 對生成結果的影響
4. **微調對比**: 對比預訓練模型和微調模型的差異
5. **實際應用**: 討論語言模型在不同場景的應用

[技術細節]
生成流程:
1. 輸入文字經過 tokenizer 轉換為 token IDs
2. 模型根據 token IDs 預測下一個 token 的機率分佈
3. 根據採樣策略（temperature, top-p, top-k）選擇下一個 token
4. 重複步驟 2-3 直到生成結束符號或達到最大長度
5. 將 token IDs 解碼回文字

[檔案結構]
gradio.py                        # 本檔案 - Gradio 展示介面
model.safetensors                # 模型權重檔案
config.json                      # 模型配置檔案
tokenizer.json                   # 分詞器配置
special_tokens_map.json          # 特殊 token 對應

[注意事項]
- 首次載入模型需要一些時間
- CPU 運行速度較慢，建議使用 GPU
- 模型檔案較大 (~800MB)，確保有足夠儲存空間
- 預訓練模型的回答品質依賴輸入 prompt 的設計

[後續進階]
這個展示是基礎版本，可進一步：
1. 進行任務特定微調（對話、問答、摘要等）
2. 使用 QLoRA 進行參數高效微調
3. 加入 prompt engineering 技巧
4. 整合外部知識庫（RAG）
5. 多輪對話記憶管理

[開發者]
碩士班課程專案 - 深度學習（進階）
建立日期: 2024
更新日期: 2026-03-10 (修正 Gradio 6.0 相容性)

===============================================================================
"""

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
    demo.launch(share=False, server_port=7860)
