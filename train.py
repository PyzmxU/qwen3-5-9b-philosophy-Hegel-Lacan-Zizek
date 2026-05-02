import unsloth
import os
import config  # 加载代理设置（side-effect: 设置环境变量）
from transformers import BitsAndBytesConfig
# 开启 PyTorch 动态显存碎片回收
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# 👉 新增以下三行：物理切断 Hugging Face 和 Unsloth 的联网遥测，实现秒启动
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["UNSLOTH_OFFLINE"] = "1"


# 👉 🚨 核心修复：把数据集和模型的缓存全部重定向到当前目录下的 cache 文件夹
os.environ["HF_DATASETS_CACHE"] = "./hf_cache/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "./hf_cache/hub"

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template

from data_handler import load_and_prepare_dataset

def main():
    print(f"正在以‘极限压制’模式装载 4-bit 模型: {config.MODEL_NAME_OR_PATH}...")
# 🚨 强制 BNB 配置：不再给模型任何回退到 BF16 的机会
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",          # 使用更精准的规格
        bnb_4bit_compute_dtype = torch.bfloat16,
        bnb_4bit_use_double_quant = True,     # 开启双量化，再省下约 1GB 显存
    )




    print(f"正在装载 4-bit 模型: {config.MODEL_NAME_OR_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.MODEL_NAME_OR_PATH,
        max_seq_length = config.MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = config.LOAD_IN_4BIT,
        # attn_implementation = "flash_attention_2",
        # quantization_config = bnb_config,     # 🚀 注入硬核配置


        attn_implementation = "sdpa",         # 🚨 🚨 核心修改：必须改回 sdpa！因为你的 FA2 安装失败了，强行写 FA2 会导致加载时的额外显存开销
        # device_map = {"": 0},                 # 🚨 强制指定单卡 0，防止参数漂移
        # device_map = "auto",
        # low_cpu_mem_usage = True,             # 🚨 优化 CPU 到 GPU 的传输逻辑
    )

    # 强制绑定 ChatML 模板 (必须取消注释)
    tokenizer = get_chat_template(
        tokenizer, 
        chat_template="chatml",
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"}
    )

    # 配置全模块 LoRA
    print("正在挂载满血版 LoRA 适配器...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = config.LORA_R,
        target_modules = config.TARGET_MODULES,
        lora_alpha = config.LORA_ALPHA,
        lora_dropout = config.LORA_DROPOUT,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 加载数据集
    dataset = load_and_prepare_dataset(config.DATASET_ID, tokenizer, config.MAX_SEQ_LENGTH)
    actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)     

    # ==========================================
    # 🚨 终极安全校验 (Pre-flight Check)
    # ==========================================
    print("\n" + "="*50)
    print("【开训前终极检验：分词器与数据映射验证】")
    print("特殊字符映射表:", actual_tokenizer.special_tokens_map)
    print("-" * 50)
    if len(dataset) > 0:
        sample_text = dataset[0]["text"]
        print("模型即将吃入的第一条数据 (截断前800字符):\n")
        print(sample_text[:1800] + "\n...[后续已截断以供预览]")
    print("="*50 + "\n")

    # 配置 SFTTrainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = actual_tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = config.MAX_SEQ_LENGTH,
        dataset_num_proc = 8,
        packing = False, # 🚨 🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨核心修改：关闭数据打包
        args = TrainingArguments(
            output_dir = config.OUTPUT_DIR,
            per_device_train_batch_size = config.PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS,
            warmup_ratio = 0.05,            
            num_train_epochs = config.NUM_TRAIN_EPOCHS,
            learning_rate = config.LEARNING_RATE,
            bf16 = True,                    
            fp16 = False,
            logging_steps = 1,              
            optim = "adamw_8bit",           
            # optim = "paged_adamw_8bit",     
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",   
            max_grad_norm = 1.0,            
            seed = 3407,
            dataloader_num_workers = 12,       
            dataloader_prefetch_factor = 4,     
            dataloader_pin_memory = True,
            gradient_checkpointing = True,
            save_steps = 100,
            save_total_limit = 3,
            report_to = "none",             
        )
    )

    # print("🚀 开始 3090 满血微调训练...")
    print("🚀 开始 4090 48G 满血微调训练...")
    trainer.train()

    # 保存权重
    save_path = config.OUTPUT_LORA
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ LoRA 权重已成功保存至 {save_path}！")

if __name__ == "__main__":
    main()