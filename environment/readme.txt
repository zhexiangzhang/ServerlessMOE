# huggingface

pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com

sudo apt-get install aria2
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
./hfd.sh [model_name] --tool aria2c -x 8    (1-16)

git config --global user.email "zhexiangzhang@163.com"
git config --global user.name "zhexiangzhang"


# deepspeed --num_gpus 4 inference-test.py --model deepseek-ai/deepseek-moe-16b-base --checkpoint_path /root/.cache/huggingface/hub/models--deepseek-ai--deepseek-moe-16b-base/snapshots/521d2bc4fb69a3f3ae565310fcc3b65f97af2580 --batch_size 1 --dtype float16 --world_size 4 --max_new_tokens 10 --use_meta_tensor --test_performance --greedy --trust_remote_code 
# deepspeed --num_gpus 4 inference-test.py --model mistralai/Mixtral-8x7B-v0.1 --batch_size 1 --dtype float16 --world_size 4 --max_new_tokens 10 --use_meta_tensor --test_performance --greedy 
# deepspeed --num_gpus 4 inference-test.py --model deepseek-ai/deepseek-moe-16b-base --batch_size 1 --dtype float16 --world_size 4 --max_new_tokens 10 --use_meta_tensor --test_performance --greedy 