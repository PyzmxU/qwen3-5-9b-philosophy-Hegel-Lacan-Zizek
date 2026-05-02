import os
from datasets import load_dataset
import config

# 系统提示词（作为兜底：如果 JSON 数据里没有 "system" 字段，则使用这个）
PHILOSOPHER_SYSTEM_PROMPT = (
    "你是一个精通黑格尔辩证法、拉康精神分析与齐泽克意识形态批判的顶级哲学家。"
    "你擅长透过日常表象，运用三段论式结构揭示其背后的意识形态与实在界裂缝。"
)

def load_and_prepare_dataset(dataset_path, tokenizer, max_seq_length):
    print(f"📥 正在加载本地数据集: {dataset_path}...")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"找不到数据集文件: {dataset_path}")
        
    dataset = load_dataset("json", data_files=dataset_path, split="train",cache_dir="./hf_datasets_cache"  )
    
    print(f"✅ 本地数据加载成功，共 {len(dataset)} 条原始数据。")

    def format_implicit_reasoning(example):
        # 1. 提取系统提示词 (优先使用数据里自带的 "system")
        sys_content = example.get("system", "")
        if not sys_content:
            sys_content = PHILOSOPHER_SYSTEM_PROMPT

        # 2. 提取并拼接用户提问 (instruction + input)
        instruction = example.get("instruction", "").strip()
        input_text = example.get("input", "").strip()
        
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction

        # 3. 提取模型回答
        assistant_content = example.get("output", "").strip()

        # 异常数据拦截
        if not user_content or not assistant_content:
            return {"text": "", "token_length": 0}

        # 4. 构造标准消息结构
        formatted_messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        
        try:
            # 5. 调用 Tokenizer 的模板进行渲染 (ChatML 或 Qwen 原生模板)
            text = tokenizer.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        except Exception as e:
            print(f"⚠️ 模板渲染失败: {e}")
            text = ""
            
        # Token 长度预估 (仅作截断过滤参考)
        estimated_tokens = len(text) // 2
        return {"text": text, "token_length": estimated_tokens}

    print("🚀 正在执行并行模板映射...")
    processed_dataset = dataset.map(format_implicit_reasoning, num_proc=8)

    # 6. 清洗数据：剔除空数据和超长数据
    final_dataset = processed_dataset.filter(lambda x: 0 < x["token_length"] < max_seq_length)
    
    # 7. 移除原始列，只保留给模型吃的 "text" 字段
    columns_to_remove = [col for col in final_dataset.column_names if col != "text"]
    final_dataset = final_dataset.remove_columns(columns_to_remove)

    print(f"🧹 清洗完毕：保留 {len(final_dataset)} 条有效数据。")
    return final_dataset.shuffle(seed=3407)