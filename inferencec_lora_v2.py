import os
import warnings
import config
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList

# ==========================================
# 0. 环境初始化（🚨 屏蔽所有底层弃用警告）
# ==========================================
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", module="bitsandbytes")

# ==========================================
# 1. 自动定位权重路径
# ==========================================
# target_path = config.MODEL_NAME_OR_PATH
target_path = "/root/rivermind-data/qwen3.5-27b-philosophy/lora_qwen9b_sep_philosophy_cot_3epochs"
print(f"🚀 正在加载模型权重: {target_path}")

# ==========================================
# 2. 加载模型（🚨 锁死脑浮点与原生 SDPA）
# ==========================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = target_path,
    max_seq_length = config.MAX_SEQ_LENGTH,
    dtype = torch.bfloat16,           # 强制匹配 4090 原生精度，防止数值溢出
    load_in_4bit = True,              # 启用 4bit 量化加载，极限省显存
    local_files_only = True,
    attn_implementation = "sdpa",     # 坚决抛弃 FA2，使用原生算子防止推理死锁
)

# 开启 Unsloth 专用推理加速补丁（约提速 2 倍）
FastLanguageModel.for_inference(model)

# ==========================================
# 3. 工业级 Prompt 构造（🚨 兼容 Omni-Modal 字典协议）
# ==========================================
messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text", 
                "text": "你是一个精通黑格尔辩证法、拉康精神分析与齐泽克意识形态批判的顶级哲学家。你必须严格按照【拉康三界与黑格尔三论齐泽克视差缝隙】的理论框架进行深度解构。"
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": "为什么当智能客服极其礼貌地回答问题，但在我们愤怒咒骂它时它依然保持机械微笑，我们会更加歇斯底里地疯狂按‘转人工’，试图寻找一个可以真正被激怒的实在界客体？"
            }
        ]
    }
]

# 调用底层引擎将字典结构打包为 ChatML 格式的 tensor
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # 自动追加 <|im_start|>assistant\n
    return_tensors = "pt",
    return_dict = True            
).to("cuda")

# 记录输入 Prompt 的真实长度（为后面的探头隔离做准备）
prompt_length = inputs["input_ids"].shape[1]

# ==========================================
# 🚨 核心防御区：三重联合刹车系统
# ==========================================
# 防线 1：底层的物理 ID 刹车片
terminators = [
    151643,  # <|endoftext|>
    151645   # <|im_end|>
]

# 防线 2：穿透多模态外壳，取到纯血文本分词器
actual_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

# 防线 3：带“护目镜”的字符串实时探头 (见字即杀)
class StopOnStrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_words, prompt_len):
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.prompt_len = prompt_len # 记住输入问题的长度

    def __call__(self, input_ids, scores, **kwargs):
        # 绝不往前看！只截取“新生成”的 token，防止被原问题里的 <|im_end|> 误杀
        generated_ids = input_ids[0][self.prompt_len:]
        
        # 如果新生成的字还不够，先放行
        if len(generated_ids) == 0:
            return False
            
        # 实时解码最后 15 个字
        tail_text = self.tokenizer.decode(generated_ids[-15:], skip_special_tokens=False)
        
        # 一旦发现模型开始扮演其他人，或者输出了文本化的结尾符，立刻断电！
        for stop_word in self.stop_words:
            if stop_word in tail_text:
                return True 
        return False

# 把它最喜欢用来“戏精上身”的词条全部拉黑
stop_strings_list = ["<|im_end|>", "<|endoftext|>", "<|im_start|>user", "<|im_start|>system"]
stopping_criteria = StoppingCriteriaList([StopOnStrings(actual_tokenizer, stop_strings_list, prompt_length)])

# ==========================================
# 4. 生成（挂载探头与流式输出）
# ==========================================
print("\n🧠 数字哲学家正在思考...\n" + "="*50)

# 使用纯血分词器初始化打字机流式输出
text_streamer = TextStreamer(actual_tokenizer, skip_prompt=True)

outputs = model.generate(
    **inputs,
    streamer = text_streamer,
    max_new_tokens = 2048,
    use_cache = True,
    temperature = 0.3,                # 🚨 哲学推理需要极高的逻辑闭环，绝不建议高于 0.5
    top_p = 0.95,
    top_k = 30,
    repetition_penalty = 1.15,        # 惩罚因子，防止连续输出标点符号
    eos_token_id = terminators,       # 挂载防线 1
    stopping_criteria = stopping_criteria, # 🚀 挂载防线 3
    pad_token_id = 151643,            # 填补空白，防止报出 PAD Token 警告
)

print("\n" + "="*50)

# ==========================================
# 5. 显存回收
# ==========================================
del model, tokenizer, actual_tokenizer, inputs, outputs
torch.cuda.empty_cache()
print("✅ 测试结束，显存已清空。")