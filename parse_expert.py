import os
import matplotlib.pyplot as plt

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
                
        tail_10 = lines[-10:]
        
        time_ = []  
        cnt_ = [] 
                
        for line in tail_10:
            parts = line.strip().split(',')
            cnt = int(parts[0].split('=')[1])
            time = float(parts[1].split('=')[1])
            cnt_.append(cnt)
            time_.append(time)
                
        avg_time_all = sum(time_) / len(time_)
        avg_cnt_all = sum(cnt_) / len(cnt_)        
                
        return (avg_cnt_all, avg_time_all)

def plot_results(results):
    batch_sizes = sorted([int(name.split('_')[1]) for name in results.keys()])
    delays = [results[f"batch_{size}_new_tokens_6.txt"][0] for size in batch_sizes]
    avg_atts_all = [results[f"batch_{size}_new_tokens_6.txt"][1] for size in batch_sizes]
    avg_moes_all = [results[f"batch_{size}_new_tokens_6.txt"][2] for size in batch_sizes]
    avg_atts_first_token = [results[f"batch_{size}_new_tokens_6.txt"][3] for size in batch_sizes]
    avg_moes_first_token = [results[f"batch_{size}_new_tokens_6.txt"][4] for size in batch_sizes]

    # 图1：端到端延迟
    plt.figure(figsize=(10, 5))
    plt.plot(batch_sizes, delays, marker='o')
    plt.title("End-to-End Latency")
    plt.xlabel("Batch Size")
    plt.ylabel("Delay (s)")
    plt.grid(True)
    plt.xticks(batch_sizes)
    plt.savefig("end_to_end_delays_2.png")

    # 图2：ATT和MOE平均值
    plt.figure(figsize=(10, 5))
    plt.plot(batch_sizes, avg_atts_all, marker='o', label='Attention Layer (Decode)', color = 'b')
    plt.plot(batch_sizes, avg_moes_all, marker='o', label='MOE Layer (Decode)', linestyle = 'dotted')
    plt.plot(batch_sizes, avg_atts_first_token, marker='o', label='Attention Layer (Prefill)', color = 'r') 
    plt.plot(batch_sizes, avg_moes_first_token, marker='o', label='MOE Layer (Prefill )', linestyle = 'dotted', color = 'orangered')
    plt.title("Average Attention Layer and MOE Layer Times")
    plt.xlabel("Batch Size")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.xticks(batch_sizes)
    plt.savefig("att_moe_averages_2.png")


def main(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            results[filename] = process_file(file_path)
    
    sort_result = {}
    # 打印结果
    for filename, data in results.items():
        print(f"{filename}:")
        print(f" cnt : {int(data[0])}")        
        time = round(data[1], 2)
        print(f" time : {time}")
    
        sort_result[int(data[0])] = time
        sort_result = dict(sorted(sort_result.items(), key=lambda item: item[0]))
    
    batch_list = list(sort_result.keys())
    time_list = list(sort_result.values())
    
    print(time_list)
    # plot_results(results)

    plt.figure(figsize=(10, 5))
    # 将数值标签到折线上
    
    for a, b in zip(batch_list, time_list):
        # 数值在点的上方2个单位位置
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)    

    plt.plot(batch_list, time_list, marker='o')
    plt.title("Single Expert Latency")
    plt.xlabel("Batch Size")
    plt.ylabel("Delay (ms)")
    plt.grid(True)
    plt.xticks(batch_list)
    plt.savefig("single_expert.png")

if __name__ == "__main__":
    directory = 'measure_data_expert'  
    main(directory)
