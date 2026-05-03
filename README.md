# 🧠 Qwen3.5-9B Philosophy Scripts

本仓库是 **Qwen3.5-9B-Philosophy-Hegel-Lacan-Zizek** 的脚本仓库，主要保存训练、数据处理、推理、GGUF 导出和 ModelScope 上传脚本。

📦 **模型权重与 GGUF 文件请前往魔塔 ModelScope：**  
https://www.modelscope.cn/models/oooooo0o/qwen3-5-9b-philosophy-Hegel-Lacan-Zizek-qlora

💾 **GGUF 目录：**  
https://www.modelscope.cn/models/oooooo0o/qwen3-5-9b-philosophy-Hegel-Lacan-Zizek-qlora/tree/master/gguf

> GitHub 只保存代码和配置，不保存模型权重、LoRA 适配器、GGUF 文件、训练 checkpoint、缓存和本地数据集，避免超大文件与凭据泄漏。

## ✨ 项目说明

该项目基于 **Qwen3.5-9B** 做哲学领域 SFT 微调，主题聚焦：

- 🏛️ 黑格尔辩证法
- 🧩 拉康精神分析
- 🔍 齐泽克意识形态批判
- 🧠 长链路哲学推理

模型成品、LoRA、safetensors 和 GGUF 量化版本均托管在魔塔 ModelScope。

## 🗂️ 脚本结构

```text
config.py                 # ⚙️ 训练路径、模型路径和超参数
data_handler.py           # 🧾 SFT 数据格式化与样本处理
train.py                  # 🚂 Unsloth LoRA SFT 训练入口
inferencec_lora_v2.py     # 💬 本地 LoRA 推理与 smoke test
export_gguf.py            # 📦 合并权重并导出 GGUF
download_model.py         # ⬇️ ModelScope 基座模型下载辅助脚本
check_model.py            # 🔎 ModelScope 模型下载/检查辅助脚本
upload_modelscope.py      # ⬆️ ModelScope 产物上传辅助脚本
ds_zero2.json             # 🧱 DeepSpeed ZeRO-2 配置
ds_zero3.json             # 🧱 DeepSpeed ZeRO-3 配置
requirements.txt          # 📌 Python 依赖列表
```

训练和导出后可能生成的本地产物：

```text
lora*/          # LoRA adapters
safetensors*/   # merged BF16 weights
gguf*/          # GGUF quantized models
outputs*/       # trainer checkpoints
```

这些路径已在 `.gitignore` 中排除。

## 🚀 快速开始

```bash
pip install -r requirements.txt
python download_model.py
python train.py
python inferencec_lora_v2.py
```

## 🧪 推理测试

```bash
python inferencec_lora_v2.py
```

脚本默认用于加载本地 LoRA 产物并进行一次哲学问答推理测试。实际模型路径、LoRA 路径和生成参数请按本机环境修改 `config.py` 或脚本内配置。

## 📦 导出 GGUF

```bash
python export_gguf.py
```

导出后的 GGUF 文件建议上传到魔塔仓库：

```text
gguf/Qwen3.5-9B.BF16-mmproj.gguf
gguf/Qwen3.5-9B.Q4_K_M.gguf
gguf/Qwen3.5-9B.Q5_K_M.gguf
gguf/Qwen3.5-9B.Q6_K.gguf
gguf/Qwen3.5-9B.Q8_0.gguf
```

## ⬆️ 上传到 ModelScope

不要把 token 写入脚本或提交到 GitHub：

```bash
export MODELSCOPE_TOKEN="your_token"
python upload_modelscope.py
```

目标魔塔仓库：

```text
oooooo0o/qwen3-5-9b-philosophy-Hegel-Lacan-Zizek-qlora
```

## 🧰 训练配置概览

| 参数 | 值 |
|------|------|
| 🧬 微调方式 | LoRA 全模块适配器 |
| 🎯 目标模块 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 📏 最大序列长度 | 8192 |
| 🔢 LoRA rank | 32 |
| ⚖️ LoRA alpha | 64 |
| 📉 学习率 | 2e-5 |
| 🧮 优化器 | adamw_8bit |
| 🌊 调度器 | cosine |
| ⚡ 注意力实现 | sdpa |

## 🔗 相关链接

- 📦 ModelScope 模型主页：  
  https://www.modelscope.cn/models/oooooo0o/qwen3-5-9b-philosophy-Hegel-Lacan-Zizek-qlora
- 💾 ModelScope GGUF 文件夹：  
  https://www.modelscope.cn/models/oooooo0o/qwen3-5-9b-philosophy-Hegel-Lacan-Zizek-qlora/tree/master/gguf
- 🧑‍💻 GitHub 脚本仓库：  
  https://github.com/PyzmxU/qwen3-5-9b-philosophy-Hegel-Lacan-Zizek

## ⚠️ 注意

- 🔐 不要提交 ModelScope token、GitHub token 或本地凭据文件。
- 🧱 不要把大模型权重、GGUF、checkpoint 直接提交到 GitHub。
- 🌡️ 推理温度建议不高于 `0.5`，避免哲学推理链条漂移。
