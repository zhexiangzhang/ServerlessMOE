import os
import matplotlib.pyplot as plt

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # end to end delay
        end_to_end_delay = float(lines[-1].strip())
        
        att_values = []
        moe_values = []
                
        for line in lines[:192]:    mixtral
        # for line in lines[:168]:        
            parts = line.strip().split(',')
            print(parts)
            att = float(parts[0].split('=')[1])
            moe = float(parts[1].split('=')[1])
            att_values.append(att)
            moe_values.append(moe)
                
        avg_att_all = sum(att_values) / len(att_values)
        avg_moe_all = sum(moe_values) / len(moe_values)
        
        # avg_att_first_token = sum(att_values[:28]) / 28
        # avg_moe_first_token = sum(moe_values[:28]) / 28
        avg_att_first_token = sum(att_values[:32]) / 32
        avg_moe_first_token = sum(moe_values[:32]) / 32

        return (end_to_end_delay, avg_att_all, avg_moe_all, avg_att_first_token, avg_moe_first_token)

def plot_results(results):
    batch_sizes = sorted([int(name.split('_')[1]) for name in results.keys()])
    delays = [results[f"(llama) batch_{size}_new_tokens_6.txt"][0] for size in batch_sizes]
    avg_atts_all = [results[f"(llama) batch_{size}_new_tokens_6.txt"][1] for size in batch_sizes]
    avg_moes_all = [results[f"(llama) batch_{size}_new_tokens_6.txt"][2] for size in batch_sizes]
    avg_atts_first_token = [results[f"(llama) batch_{size}_new_tokens_6.txt"][3] for size in batch_sizes]
    avg_moes_first_token = [results[f"(llama) batch_{size}_new_tokens_6.txt"][4] for size in batch_sizes]

    # 图1：端到端延迟
    plt.figure(figsize=(10, 5))
    # 把数值画出来
    for i in range(len(batch_sizes)):
        plt.text(batch_sizes[i], delays[i], f"{delays[i]:.2f}", ha='center', va='bottom', fontsize=8)        
    plt.plot(batch_sizes, delays, marker='o')
    plt.title("[Llama] End-to-End Latency")
    plt.xlabel("Batch Size")
    plt.ylabel("Delay (s)")
    plt.grid(True)
    plt.xticks(batch_sizes)
    plt.savefig("[Llama] end_to_end_delays_2.png")

    # 图2：ATT和MOE平均值
    plt.figure(figsize=(10, 5))
    plt.plot(batch_sizes, avg_atts_all, marker='o', label='Attention Layer (Decode)', color = 'b')
    plt.plot(batch_sizes, avg_moes_all, marker='o', label='MOE Layer (Decode)', linestyle = 'dotted')
    plt.plot(batch_sizes, avg_atts_first_token, marker='o', label='Attention Layer (Prefill)', color = 'r') 
    plt.plot(batch_sizes, avg_moes_first_token, marker='o', label='MOE Layer (Prefill )', linestyle = 'dotted', color = 'orangered')
    # plt.plot(batch_sizes, [0.66, 3.31, 3.32, 3.53, 3.59, 3.62, 4.49, 5.65, 8.69], marker='o', label='Reference', linestyle = 'dotted', color = 'black')
    plt.title("[Llama] Average Attention Layer and MOE Layer Times")
    plt.xlabel("Batch Size")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.xticks(batch_sizes)
    plt.savefig("[Llama] att_moe_averages_2.png")


def main(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            results[filename] = process_file(file_path)
    
    # 打印结果
    for filename, data in results.items():
        print(f"{filename}:")
        print(f"  End-to-End Delay: {data[0]}s")
        print(f"  Average ATT (all tokens): {data[1]:.2f}")
        print(f"  Average MOE (all tokens): {data[2]:.2f}")
        print(f"  Average ATT (first token): {data[3]:.2f}")
        print(f"  Average MOE (first token): {data[4]:.2f}")
        print()

    plot_results(results)

if __name__ == "__main__":
    # [0.66, 3.31, 3.32, 3.53, 3.59, 3.62, 4.49, 5.65, 8.69]
    directory = 'measure_data_llama_deepseek'
    main(directory)
