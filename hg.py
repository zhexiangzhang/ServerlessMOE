import time, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "deepseek-ai/deepseek-moe-16b-base"
# flash_attn!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
precision = torch.float16
device_map_auto = "auto"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# deepseek-ai/deepseek-moe-16b-base
model = None
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token # to avoid an error

text16 = "In the deep blue sky, a small bird sings joyfully on a high branch"
repeat = "Beside the foggy docks, seagulls perch stoically, unfazed by the ebb and flow of tides"
# repeat = "He is a boy"

def initModel(device_map, print_device_map = False):
    global model
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto", trust_remote_code=True)
    if print_device_map:
        print(model.hf_device_map)    

# cold start
def coldStart():
    print("-------- cold start --------")
    inputs = tokenizer(text16, return_tensors="pt").to(0)
    outputs = model.generate(**inputs, max_new_tokens=2, min_new_tokens=2)
    runExp(repeats, "batching", 4)
    runExp(repeats, "batching", 4)
    print("------- cold start end -------")
    print()    
    print()
    print()

def runExp(text, input_tokens, output_tokens, post_process = False):
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(0)
    # inputs = tokenizer(text, return_tensors="pt").to(0)
    time0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=output_tokens, min_new_tokens=output_tokens)
    time1 = time.time()
    elapsed = time1 - time0
    print("[{}-{}]= {:.2f}s".format(input_tokens, output_tokens, elapsed))
    if post_process:
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(len(result))
        print(result)
    return elapsed

if __name__ == "__main__":
    import sys    
    folder_name = "measure_data_deepseek"
    tmpfile_name = "tmp_data.txt"
    if len(sys.argv) > 2:
        batch = int(sys.argv[1])
        new_tokens = int(sys.argv[2])
        print("batch size:", batch, "new tokens:", new_tokens)                
        file_name = "(deepseek) batch_" + str(batch) + "_new_tokens_" + str(new_tokens) +".txt"        
    else:
        batch = 2     
        new_tokens = 3
        # raise Exception("batch size is not provided")

    repeats = [repeat for i in range(batch)]
    print("repeats:", len(repeats))

    # experiment
    initModel("auto")
    coldStart()

    time.sleep(1)
    runExp(repeats, "batching", new_tokens)
    time.sleep(1)
    
    # 使用join
    tmpFilePath = os.path.join(folder_name, tmpfile_name)
    FilePath = os.path.join(folder_name, file_name)    
    os.mknod(tmpFilePath)
    
    end_to_end = runExp(repeats, "batching", new_tokens)
        
    os.rename(tmpFilePath, FilePath)    
    with open(FilePath, "a") as file:
        file.write(str(end_to_end))