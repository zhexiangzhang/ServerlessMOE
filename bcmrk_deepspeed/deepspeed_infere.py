import deepspeed, torch, os, time
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, MixtralForCausalLM 
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from deepspeed.runtime.utils import see_memory_usage

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# model_id = "Qwen/Qwen1.5-MoE-A2.7B"
# model_id = "mistralai/Mixtral-8x7B-v0.1"
# deepspeed --num_gpus 4 inference-test.py --model mistralai/Mixtral-8x7B-v0.1 --batch_size 1 --dtype float16 --world_size 4 --max_new_tokens 10 --use_meta_tensor --test_performance --greedy 
# deepspeed --num_gpus 4 inference-test.py --model deepseek-ai/deepseek-moe-16b-base --batch_size 1 --dtype float16 --world_size 4 --max_new_tokens 10 --use_meta_tensor --test_performance --greedy 
# deepspeed --num_gpus 4 inference-test.py --model deepseek-ai/deepseek-moe-16b-base --checkpoint_path /root/.cache/huggingface/hub/models--deepseek-ai--deepseek-moe-16b-base/snapshots/521d2bc4fb69a3f3ae565310fcc3b65f97af2580 --batch_size 1 --dtype float16 --world_size 4 --max_new_tokens 10 --use_meta_tensor --test_performance --greedy --trust_remote_code 
# deepspeed --num_gpus 4 inference-test.py --model llama-moe/LLaMA-MoE-v1-3_0B-2_16 --checkpoint_path /root/LLaMA-MoE-v1-3_5B-2_8 --batch_size 1 --dtype float16 --world_size 4 --max_new_tokens 10 --use_meta_tensor --test_performance --greedy --trust_remote_code 

config = {    
    "kernel_inject": False,
    "tensor_parallel": {"tp_size": 4},
    "dtype": "fp16"
}

model_path =  "/root/LLaMA-MoE-v1-3_5B-2_8"
# model_path = "/root/.cache/huggingface/hub/models--deepseek-ai--deepseek-moe-16b-base/snapshots/521d2bc4fb69a3f3ae565310fcc3b65f97af2580"
tokenizer = AutoTokenizer.from_pretrained(model_path,  trust_remote_code=True, use_fast = False)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,  trust_remote_code=True)

local_rank = int(os.getenv("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)

input_text = "Beside the foggy docks, seagulls perch stoically, unfazed by the ebb and flow of tides"

inputs = tokenizer.encode(input_text, return_tensors="pt").to(device=local_rank)

# hfdsc = HfDeepSpeedConfig(ds_config)

# model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
# deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
# model.eval()

see_memory_usage("After load model", force=True)  
# MA 0.0 GB         Max_MA 0.0 GB         CA 0.0 GB         Max_CA 0 GB 
# CPU Virtual Memory:  used = 14.58 GB, percent = 4.0%
ds_engine = deepspeed.init_inference(model=model, config=config)
ds_engine.module.eval()
model = ds_engine.module
see_memory_usage("After DS-inference init", force=True)
# MA 22.62 GB         Max_MA 22.62 GB         CA 22.74 GB         Max_CA 23 GB
# CPU Virtual Memory:  used = 42.89 GB, percent = 11.9%

torch.cuda.synchronize()
start = time.time()
gen_tokens = model.generate(
        inputs,        
        # do_sample=True,
        # temperature=0.5,
        min_new_tokens=6,
        max_new_tokens=6,
        synced_gpus=True
    )
torch.cuda.synchronize()
end = time.time()

see_memory_usage("After forward", force=True)
# MA 22.62 GB         Max_MA 22.63 GB         CA 22.76 GB         Max_CA 23 GB
# CPU Virtual Memory:  used = 43.13 GB, percent = 11.9%

gen_text = tokenizer.decode(gen_tokens[0])

print(gen_text)
print("Inference Time:", end - start)  #  2.285



# [2024-06-14 13:13:31,347] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.14.3, git-hash=unknown, git-branch=unknown
# [2024-06-14 13:13:31,348] [INFO] [logging.py:96:log_dist] [Rank -1] quantize_bits = 8 mlp_extra_grouping = False, quantize_groups = 1
# [2024-06-14 13:13:31,348] [INFO] [logging.py:96:log_dist] [Rank -1] quantize_bits = 8 mlp_extra_grouping = False, quantize_groups = 1
# [2024-06-14 13:13:31,348] [INFO] [logging.py:96:log_dist] [Rank -1] quantize_bits = 8 mlp_extra_grouping = False, quantize_groups = 1
# [2024-06-14 13:25:01,245] [INFO] [comm.py:637:init_distributed] cdb=None
# [2024-06-14 13:25:01,245] [INFO] [comm.py:637:init_distributed] cdb=None
# [2024-06-14 13:25:01,245] [INFO] [comm.py:637:init_distributed] cdb=None
# [2024-06-14 13:25:01,245] [INFO] [comm.py:637:init_distributed] cdb=None
# [2024-06-14 13:25:01,257] [INFO] [comm.py:668:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
# AutoTP: AutoTP:  AutoTP:  [(<class 'transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer'>, ['6.w2', 'self_attn.o_proj', '5.w2', '2.w2', '7.w2', '4.w2', '1.w2', '0.w2', '3.w2'])] 
# [(<class 'transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer'>, ['3.w2', '6.w2', '0.w2', '7.w2', 'self_attn.o_proj', '2.w2', '1.w2', '4.w2', '5.w2'])][(<class 'transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer'>, ['5.w2', '4.w2', '7.w2', '0.w2', '2.w2', '3.w2', '6.w2', '1.w2', 'self_attn.o_proj'])]






# Loading checkpoint shards: 100%|██████████████████████████| 19/19 [00:03<00:00,  4.79it/s]
# Loading checkpoint shards: 100%|██████████████████████████| 19/19 [00:04<00:00,  4.73it/s]
# Loading checkpoint shards: 100%|██████████████████████████| 19/19 [00:04<00:00,  4.70it/s]
# Loading checkpoint shards: 100%|██████████████████████████| 19/19 [00:03<00:00,  4.80it/s]
# [2024-06-14 01:40:03,602] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.14.3, git-hash=unknown, git-branch=unknown
# [2024-06-14 01:40:03,603] [INFO] [logging.py:96:log_dist] [Rank -1] quantize_bits = 8 mlp_extra_grouping = False, quantize_groups = 1
# [2024-06-14 01:40:03,644] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.14.3, git-hash=unknown, git-branch=unknown
# [2024-06-14 01:40:03,646] [INFO] [logging.py:96:log_dist] [Rank -1] quantize_bits = 8 mlp_extra_grouping = False, quantize_groups = 1
# [2024-06-14 01:40:03,693] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.14.3, git-hash=unknown, git-branch=unknown
# [2024-06-14 01:40:03,693] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.14.3, git-hash=unknown, git-branch=unknown
# [2024-06-14 01:40:03,695] [INFO] [logging.py:96:log_dist] [Rank -1] quantize_bits = 8 mlp_extra_grouping = False, quantize_groups = 1
# [2024-06-14 01:40:03,695] [INFO] [logging.py:96:log_dist] [Rank -1] quantize_bits = 8 mlp_extra_grouping = False, quantize_groups = 1
# [2024-06-14 01:51:33,831] [INFO] [comm.py:637:init_distributed] cdb=None
# [2024-06-14 01:51:33,831] [INFO] [comm.py:637:init_distributed] cdb=None
# [2024-06-14 01:51:33,831] [INFO] [comm.py:637:init_distributed] cdb=None
# [2024-06-14 01:51:33,831] [INFO] [comm.py:637:init_distributed] cdb=None
# [2024-06-14 01:51:33,844] [INFO] [comm.py:668:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
# AutoTP:  [(<class 'transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer'>, ['7.w2', 'self_attn.o_proj', '1.w2', '3.w2', '5.w2', '6.w2', '2.w2', '0.w2', '4.w2'])]
# AutoTP:  [(<class 'transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer'>, ['4.w2', '1.w2', '3.w2', '5.w2', 'self_attn.o_proj', '2.w2', '0.w2', '7.w2', '6.w2'])]
# AutoTP:  [(<class 'transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer'>, ['5.w2', '1.w2', '2.w2', '7.w2', 'self_attn.o_proj', '0.w2', '6.w2', '3.w2', '4.w2'])]
# AutoTP:  [(<class 'transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer'>, ['1.w2', '7.w2', '4.w2', '3.w2', '2.w2', 'self_attn.o_proj', '0.w2', '5.w2', '6.w2'])]
# Traceback (most recent call last):
# Traceback (most recent call last):
# Traceback (most recent call last):
# Traceback (most recent call last):
#   File "/root/fix.py", line 47, in <module>
#   File "/root/fix.py", line 47, in <module>
#   File "/root/fix.py", line 47, in <module>
#   File "/root/fix.py", line 47, in <module>
#                 gen_tokens = model.generate(gen_tokens = model.generate(gen_tokens = model.generate(gen_tokens = model.generate(



# TypeErrorTypeErrorTypeError: : TypeError: transformers.generation.utils.GenerationMixin.generate() argument after ** must be a mapping, not Tensortransformers.generation.utils.GenerationMixin.generate() argument after ** must be a mapping, not Tensor: transformers.generation.utils.GenerationMixin.generate() argument after ** must be a mapping, not Tensor

# transformers.generation.utils.GenerationMixin.generate() argument after ** must be a mapping, not Tensor

# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# [2024-06-14 01:51:54,006] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 438138
# [2024-06-14 01:51:54,018] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 438139
# [2024-06-14 01:51:54,028] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 438140
# [2024-06-14 01:51:54,029] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 438141
# [2024-06-14 01:51:54,038] [ERROR] [launch.py:325:sigkill_handler] ['/usr/bin/python3', '-u', 'fix.py', '--local_rank=3'] exits with return code = 1