import os
from modelscope import snapshot_download

# 1. 定义数据盘目标路径
target_dir = "/root/rivermind-data/modelscope_models"
os.environ["MODELSCOPE_CACHE"] = target_dir

# 2. 模型 ID (Qwen 官方原版)
# 注意：ModelScope 上的 ID 格式通常为 '组织名/模型名'
model_id = 'unsloth/Qwen3.5-9B' 

print(f"🚀 开始从 ModelScope 下载 {model_id}...")
print(f"📂 存储路径: {target_dir}")

try:
    # download_model 会自动处理断点续传
    path = snapshot_download(model_id, cache_dir=target_dir, max_workers=64)
    print(f"✅ 下载完成！模型实际路径为: {path}")
except Exception as e:
    print(f"❌ 下载失败: {e}")