# BLOOMZ-396M 中文對話模型 Demo

> 預訓練中文對話模型展示專案

## 📌 專案概述

本專案提供一個已訓練好的 BLOOMZ-396M 中文對話模型，配備完整的 Gradio 互動介面，可直接運行使用。這是一個輕量級的中文對話生成模型，適合快速測試和展示。

## 🎯 專案特色

- ✅ **即開即用** - 包含完整的模型權重，無需額外下載
- ✅ **互動介面** - 專業的 Gradio UI，支援參數調整
- ✅ **輕量級模型** - 396M 參數，推理速度快
- ✅ **中文優化** - 基於 YeungNLP/bloomz-396m-zh

## 🛠️ 技術規格

- **模型**: YeungNLP/bloomz-396m-zh
- **參數量**: 396M
- **模型大小**: 699MB (safetensors 格式)
- **精度**: float16
- **詞表大小**: 46,145

## 🚀 快速開始

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 啟動 Gradio Demo

```bash
python gradio.py
```

Demo 將在 http://localhost:7860 啟動

## 📁 檔案說明

```
pretrained-models/
├── model.safetensors          # 模型權重 (699MB)
├── config.json               # 模型配置
├── generation_config.json    # 生成參數配置
├── gradio.py                 # Gradio 介面腳本
├── requirements.txt          # Python 依賴
└── README.md                # 本文件
```

## 🎨 Gradio 介面功能

### 主要功能

- **問題輸入框** - 輸入您的問題或對話
- **可調參數** - 控制生成品質和風格
- **範例問題** - 快速測試模型能力
- **即時生成** - 按 Enter 或點擊按鈕生成回應

### 可調參數

| 參數 | 範圍 | 預設值 | 說明 |
|------|------|--------|------|
| **最大生成長度** | 32-512 | 128 | 控制回應的最大 token 數量 |
| **Temperature** | 0.1-1.5 | 0.7 | 控制創造性（越高越隨機） |
| **Top-p** | 0.1-1.0 | 0.9 | Nucleus sampling 參數 |
| **Top-k** | 10-100 | 50 | Top-k sampling 參數 |

### 範例問題

- "你好，請自我介紹一下"
- "什麼是深度學習？"
- "如何學習Python程式設計？"
- "請推薦一本好書"

## 💻 使用範例

### Python 腳本使用

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 載入模型
model_path = "."
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 生成回應
prompt = "什麼是人工智慧？"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 🔧 環境需求

### 硬體需求

- **CPU**: 任何現代 CPU (推理較慢)
- **GPU**: 建議 4GB+ VRAM (T4/RTX 2060+)
- **RAM**: 8GB+ 系統記憶體
- **儲存空間**: 約 1GB

### 軟體需求

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- Gradio 4.0+
- CUDA 11.8+ (GPU 使用)

## 📊 模型性能

### 推理速度

| 硬體 | 速度 | 備註 |
|------|------|------|
| RTX 3090 | ~50 tokens/sec | 快速 |
| T4 GPU | ~30 tokens/sec | 中等 |
| CPU (12核) | ~2-5 tokens/sec | 較慢 |

### 記憶體使用

- **GPU (FP16)**: ~800MB VRAM
- **GPU (FP32)**: ~1.5GB VRAM
- **CPU**: ~2GB RAM

## 🎯 適用場景

- ✅ 中文對話生成
- ✅ 問答系統
- ✅ 文字補全
- ✅ 簡單的知識問答
- ✅ 教學示範

## ⚠️ 限制說明

1. **知識截止日期**: 模型訓練數據截止於特定日期，可能不包含最新資訊
2. **準確性**: 生成內容僅供參考，可能包含錯誤
3. **語言**: 主要支援簡體中文，繁體支援有限
4. **模型大小**: 396M 參數相對較小，能力有限
5. **上下文長度**: 最大 2048 tokens

## 💡 使用技巧

### 獲得更好的回應

1. **清晰的提問** - 問題越具體，回應越準確
2. **調整 Temperature** - 較低值 (0.3-0.5) 獲得更確定的回應
3. **控制長度** - 根據需求調整生成長度
4. **多次嘗試** - 相同問題可能產生不同回應

### Temperature 參數建議

- **0.3-0.5**: 事實性問答、程式碼生成
- **0.6-0.8**: 一般對話、建議
- **0.9-1.2**: 創意寫作、故事生成

## 🔗 相關資源

- **基礎模型**: [YeungNLP/bloomz-396m-zh](https://huggingface.co/YeungNLP/bloomz-396m-zh)
- **BLOOM 論文**: [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100)
- **Transformers 文檔**: [HuggingFace Transformers](https://huggingface.co/docs/transformers)

## 📝 授權說明

- 模型遵循 BLOOM 的 RAIL License
- Demo 程式碼可自由使用和修改
- 請勿用於商業用途，僅供學習和研究

## 🤝 貢獻與反饋

如有任何問題或建議，歡迎提出 issue 或 pull request。

---

*模型來源: YeungNLP/bloomz-396m-zh*
*Demo 建立日期: 2024*
*用途: 教學展示與測試*
