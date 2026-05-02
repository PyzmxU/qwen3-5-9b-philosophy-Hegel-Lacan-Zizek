# Qwen3.5-9B-Philosophy-Hegel-Lacan-Zizek

基于 **Qwen3.5-9B** 的哲学领域 SFT 微调模型，专注于 **黑格尔辩证法、拉康精神分析与齐泽克意识形态批判** 的深度推理。

本 GitHub 仓库只保存训练、推理、导出和上传脚本。模型权重、LoRA 适配器、GGUF 文件、训练 checkpoint、缓存和本地数据集没有纳入 Git，以避免超大文件与凭据泄漏。

## 模型信息

| 项目 | 说明 |
|------|------|
| 基础模型 | Qwen3.5-9B（unsloth 版本） |
| 微调方式 | LoRA 全模块适配器（7 个目标模块） |
| 训练数据 | 895 条哲学领域 SFT 数据 |
| 训练轮数 | 3 epochs |
| 量化精度 | BF16（safetensors）/ Q4_K_M、Q8_0（GGUF） |
| 系统提示 | 黑格尔辩证法 + 拉康三界 + 齐泽克视差缝隙 |
| 开发者 | oooooo0o（ModelScope） |

## 仓库结构

```
config.py                 — training paths and hyperparameters
data_handler.py           — SFT dataset formatting
train.py                  — Unsloth LoRA SFT training entrypoint
inferencec_lora_v2.py     — local LoRA inference smoke test
export_gguf.py            — merge/export to GGUF with Unsloth
download_model.py         — ModelScope base model download helper
check_model.py            — ModelScope download/check helper
upload_modelscope.py      — ModelScope artifact upload helper
ds_zero2.json             — DeepSpeed ZeRO-2 config
ds_zero3.json             — DeepSpeed ZeRO-3 config
requirements.txt          — Python dependency list
```

Expected local artifacts after training/exporting:

```
lora*/          — LoRA adapters
safetensors*/   — merged BF16 weights
gguf*/          — GGUF quantized models
outputs*/       — trainer checkpoints
```

These paths are ignored by `.gitignore`.

## 快速开始

```bash
pip install -r requirements.txt
python download_model.py
python train.py
python inferencec_lora_v2.py
```

上传到 ModelScope 时不要把 token 写入脚本：

```bash
export MODELSCOPE_TOKEN="your_token"
python upload_modelscope.py
```

## 使用方式

### 方式一：加载 LoRA 适配器

使用 transformers / unsloth 加载基础模型 + LoRA 权重：

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="oooooo0o/qwen3-5-9b-philosophy-Hegel-Lacan-Zizek",
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

> LoRA 权重位于 `lora/` 目录下，需要与 Qwen3.5-9B 基础模型配合使用。

### 方式二：使用 safetensors 合并权重

`safetensors/` 包含已合并 LoRA 的完整权重（BF16 精度），共 4 个分片：

```
model.safetensors-00001-of-00004.safetensors  (~5GB)
model.safetensors-00002-of-00004.safetensors  (~5GB)
model.safetensors-00003-of-00004.safetensors  (~5GB)
model.safetensors-00004-of-00004.safetensors  (~3GB)
```

可直接作为独立模型加载，无需基础模型。

### 方式三：使用 GGUF 量化模型

`gguf/` 包含三种量化格式，适用于 **llama.cpp、Ollama、LM Studio** 等本地推理工具：

| 文件 | 量化方式 | 特点 |
|------|---------|------|
| `Qwen3.5-9B.BF16-mmproj.gguf` | BF16（半精度） | 精度最高，体积最大 |
| `Qwen3.5-9B.Q4_K_M.gguf` | 4-bit 智能量化 | 推荐本地推理，速度与精度平衡 |
| `Qwen3.5-9B.Q8_0.gguf` | 8-bit 近无损 | 哲学逻辑精度极致要求 |

## 推理示例

```python
messages = [
    {"role": "system", "content": "你是一个精通黑格尔辩证法、拉康精神分析与齐泽克意识形态批判的顶级哲学家。"},
    {"role": "user", "content": "为什么智能客服在咒骂时依然保持机械微笑？"},
]

inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to("cuda")
outputs = model.generate(
    inputs, max_new_tokens=2048, temperature=0.3, top_p=0.95, top_k=30,
    repetition_penalty=1.15
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**推荐生成参数：** `temperature=0.3`, `top_p=0.95`, `top_k=30`, `repetition_penalty=1.15`

## 训练配置

| 参数 | 值 |
|------|------|
| LoRA 秩 (r) | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0 |
| 目标模块 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 最大序列长度 | 8192 |
| 学习率 | 2e-5 |
| 优化器 | adamw_8bit |
| 调度器 | cosine |
| 注意力实现 | sdpa |

## 适用场景

- 哲学问题深度分析
- 意识形态批判
- 精神分析视角解构
- 辩证法推理

## 限制

- 本模型仅针对哲学领域微调，通用任务表现不作保证
- 推理温度不宜过高（建议 ≤ 0.5），否则逻辑链可能断裂
