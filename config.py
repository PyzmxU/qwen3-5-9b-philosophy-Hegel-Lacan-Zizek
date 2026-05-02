import os

# ==========================================
# 0. 网络加速
# ==========================================
# os.environ["HTTP_PROXY"] = "http://117.74.66.70:13128"
# os.environ["HTTPS_PROXY"] = "http://117.74.66.70:13128"
# os.environ["NO_PROXY"] = "localhost,127.0.0.1,modelscope.com,aliyuncs.com,tencentyun.com,wisemodel.cn,ebcloud.com,tsinghua.edu.cn,pypi.org,files.pythonhosted.org,mirrors.aliyun.com,archive.ubuntu.com,security.ubuntu.com,ebtech.com"

# ==========================================
# 1. 路径与模型配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🚨 替换为官方原版模型，而非 unsloth 专属版
#MODEL_NAME_OR_PATH = "Qwen/Qwen3.5-27B" 
#DATASET_ID = "bingbangboom/philosophia-QA" 
DATASET_ID = "/root/rivermind-data/qwen3.5-27b-philosophy/philosophy_sft_v1_895.jsonl"
MODEL_NAME_OR_PATH = "/root/rivermind-data/modelscope_models/unsloth/Qwen3.5-9B"

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_qwen9b_sep_philosophy")
OUTPUT_LORA = os.path.join(BASE_DIR, "lora_qwen9b_sep_philosophy_cot")
# DS_CONFIG_PATH = os.path.join(BASE_DIR, "ds_zero2.json") # 新增 Deepspeed 配置
# DS_CONFIG_PATH = os.path.join(BASE_DIR, "ds_zero3.json")

# ==========================================
# 2. 算力保命区：双 4090 显存配置
# ==========================================
MAX_SEQ_LENGTH = 8192     # 双卡加持下，9B模型跑8K上下文绰绰有余
LOAD_IN_4BIT = True

# ==========================================
# 3. 智商解封：LoRA 全模块与高秩配置
# ==========================================
LORA_R = 32
LORA_ALPHA = 64
# LORA_DROPOUT = 0.05        # 稍微给一点 dropout 防止过拟合
LORA_DROPOUT = 0        # 稍微给一点 dropout 防止过拟合
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ==========================================
# 4. 训练节奏调度
# ==========================================
PER_DEVICE_TRAIN_BATCH_SIZE = 16       # 双卡总 Batch Size = 2 * 2(卡) * 2(累积) = 8
GRADIENT_ACCUMULATION_STEPS = 1       
NUM_TRAIN_EPOCHS = 3       
LEARNING_RATE = 2e-5                  
# OPTIMIZER = "paged_adamw_32bit"       # 配合 QLoRA 使用的分页优化器
OPTIMIZER = "adamw_8bit" # 在 DS 模式下，这个设置会被 DS 配置文件覆盖