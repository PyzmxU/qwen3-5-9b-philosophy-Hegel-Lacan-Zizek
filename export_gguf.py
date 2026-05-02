import os
import config  # 确保这里加载的是 9B 项目的 config.py

# ==========================================
# 0. 环境与网络防线
# ==========================================
os.environ["UNSLOTH_DISABLE_TELEMETRY"] = "1"

from unsloth import FastLanguageModel
import torch

# ==========================================
# 1. 路径配置
# ==========================================
# 自动读取你刚刚训练完保存的 LoRA 路径
lora_dir = "/root/rivermind-data/qwen3.5-27b-philosophy/lora_qwen9b_sep_philosophy_cot_3epochs"
export_dir = os.path.join(config.BASE_DIR, "lora_qwen9b_sep_philosophy_cot_3epochs_GGUF")
os.makedirs(export_dir, exist_ok=True)

# ==========================================
# 2. 加载模型进行合并 (Merge)
# ==========================================
print(f"📥 正在加载 9B 基础模型并合并 LoRA 权重: {lora_dir}")
# 注意：导出 GGUF 时，Unsloth 会在后台将 LoRA 权重与 Base 模型进行数学合并
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = lora_dir, 
    max_seq_length = config.MAX_SEQ_LENGTH,
    dtype = torch.bfloat16, # 使用 bf16 保证精度不丢失
    load_in_4bit = False,   # 合并权重必须设为 False
    local_files_only = True,
)

# ==========================================
# 3. 执行 GGUF 导出 (量化压缩)
# ==========================================
print("\n⚙️ 正在执行量化并导出为 GGUF 格式...")

# 方案 A：q4_k_m (最平衡，推荐用于本地 3090 或手机/Mac 推理)
# 它会智能地对关键权重保留更高精度，对不重要的层进行 4bit 压缩
print("-> 正在生成 q4_k_m 版本...")
model.save_pretrained_gguf(
    export_dir, 
    tokenizer, 
    quantization_method = "q4_k_m",
)

# 方案 B：q8_0 (近乎无损，适合对哲学逻辑精度有极致要求的场景)
print("-> 正在生成 q8_0 版本...")
model.save_pretrained_gguf(
    export_dir, 
    tokenizer, 
    quantization_method = "q8_0",
)

print(f"\n✅ 导出成功！")
print(f"路径: {export_dir}")
print(f"你可以直接用 LM Studio, Ollama 或 llama.cpp 加载这些文件了。")


# # import os
# # from unsloth import FastLanguageModel

# # # ==========================================
# # # 1. 路径配置
# # # ==========================================
# # # 指向你昨晚跑出来的最终 LoRA 文件夹
# # lora_dir = "/root/qwen3.5_4b_opus/lora_qwen3.5_4b_cot"
# # # GGUF 文件的输出目录
# # export_dir = "/root/qwen3.5_4b_opus/gguf_exports"

# # os.makedirs(export_dir, exist_ok=True)

# # # ==========================================
# # # 2. 加载模型与权重
# # # ==========================================
# # print(f"📥 正在加载基础模型与 LoRA 适配器: {lora_dir}")
# # model, tokenizer = FastLanguageModel.from_pretrained(
# #     model_name = lora_dir, 
# #     max_seq_length = 16384,
# #     dtype = None,
# #     load_in_4bit = False, # 导出时必须关闭 4bit，以全精度读取
# # )

# # # ==========================================
# # # 3. 导出 GGUF (llama.cpp 格式)
# # # ==========================================
# # print("\n⚙️ 开始编译并合并为 GGUF 格式...")

# # # 导出选项 A：q4_k_m (最推荐的平衡量化方案)
# # # 兼顾了推理速度和模型智商，体积约为 16bit 的四分之一
# # model.save_pretrained_gguf(
# #     export_dir, 
# #     tokenizer, 
# #     quantization_method = "q4_k_m",
# # )

# # # 导出选项 B：q8_0 (近乎无损的量化方案)
# # # 如果你想最大程度保留 Claude 的思维链逻辑，推荐加上这个
# # model.save_pretrained_gguf(
# #     export_dir, 
# #     tokenizer, 
# #     quantization_method = "q8_0",
# # )

# # print(f"\n✅ 恭喜！GGUF 模型已成功导出至: {export_dir}")


# import os
# import config  # 加载代理设置（side-effect: 设置环境变量）

# # ==========================================
# # 0. 网络防线 (必须放在 unsloth 导入之前)
# # ==========================================
# os.environ["UNSLOTH_DISABLE_TELEMETRY"] = "1"

# from unsloth import FastLanguageModel

# # ==========================================
# # 1. 路径配置
# # ==========================================
# lora_dir = config.OUTPUT_LORA
# export_dir = os.path.join(config.BASE_DIR, "gguf_exports")
# os.makedirs(export_dir, exist_ok=True)

# # ==========================================
# # 2. 加载模型与权重
# # ==========================================
# print(f"📥 正在加载基础模型与 LoRA 适配器: {lora_dir}")
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = lora_dir, 
#     max_seq_length = 16384,
#     dtype = None,
#     load_in_4bit = False, 
#     local_files_only = True, # 强制严禁联网检查基础模型
# )

# # ==========================================
# # 3. 导出 GGUF (llama.cpp 格式)
# # ==========================================
# print("\n⚙️ 开始编译并合并为 GGUF 格式...")

# # 导出 q4_k_m 格式
# model.save_pretrained_gguf(
#     export_dir, 
#     tokenizer, 
#     quantization_method = "q4_k_m",
# )

# # 导出 q8_0 格式
# model.save_pretrained_gguf(
#     export_dir, 
#     tokenizer, 
#     quantization_method = "q8_0",
# )

# print(f"\n✅ 恭喜！GGUF 模型已成功导出至: {export_dir}")