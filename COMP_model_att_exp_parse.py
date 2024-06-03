import os
import matplotlib.pyplot as plt

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # end to end delay
        end_to_end_delay = float(lines[-1].strip())
        
        att_values = []        
                
        # for line in lines[36:192]:    #mixtral, llama
        # for line in lines[31:168]: 
        for line in lines[26:144]:   # qwen     
            parts = line.strip().split(',')            
            att = float(parts[0].split('=')[1])            
            att_values.append(att)            
                
        avg_att_all = sum(att_values) / len(att_values)        

        # 保留3位小数
        avg_att_all = round(avg_att_all, 3)

        return avg_att_all

def process_expert_file(file_path):
    with open(file_path, 'r') as file:    
        lines = file.readlines()
        if len(lines) < 1:
            return (128, 0)        
        tail_10 = lines[-11:-1]
        
        time_ = []  
        cnt_ = 0
                
        for line in tail_10:
            parts = line.strip().split(',')
            # print(parts)
            cnt = int(parts[0].split('=')[1])
            time = float(parts[1].split('=')[1])            
            time_.append(time)
                
        avg_time_all = sum(time_) / len(time_)        

        # 保留3位小数
        avg_time_all = round(avg_time_all, 3)
        return (cnt, avg_time_all)
    
def main(directory, title):
    results = {}
    exxpert_results = {}    
    for filename in os.listdir(directory):    
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)                        
            batch = int(filename.split('_')[1])         
            # print(filename, results[filename], batch)    
            results[batch] = process_file(file_path)      
    # 按照batch排序
    results = results.items()
    results = sorted(results, key=lambda x: x[0])
    print(results)
    
    expert_directory = directory + '_expert'
    for filename in os.listdir(expert_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(expert_directory, filename)
            exxpert_results[filename] = process_expert_file(file_path)

    # print result
    # for key, value in results.items():
    #     print(key, value)

    expert_dict = {}
    for key, (x,y) in exxpert_results.items():
        # print(x, y)
        if x == 128:
            continue
        expert_dict[x] = y
    # 按照x排序
    expert_dir = expert_dict.items()
    expert_dir = sorted(expert_dir, key=lambda x: x[0])
    print(expert_dir)

    # plot
    batch_sizes = [x[0] for x in results]
    att_values = [x[1] for x in results]
    expert_values = [x[1] for x in expert_dir]
    expert_keys = [x[0] for x in expert_dir]
    # print(batch_sizes, att_values)
    # print(expert_keys, expert_values)
    plt.figure(figsize=(10, 5))
    # 从16开始表数值，数值高于点1cm
    # import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms

    for i in range(len(batch_sizes)):
        if i >= 4:
            offset = mtransforms.ScaledTranslation(0, 0.1 / 2.54, plt.gcf().dpi_scale_trans)  # 0.1 cm / 2.54 to convert cm to inches
            transform = plt.gca().transData + offset
            plt.text(batch_sizes[i], att_values[i], f"{att_values[i]:.2f}", ha='center', va='bottom', fontsize=8, transform=transform)
    
    plt.plot(batch_sizes, att_values, marker='o', label='Attention Layer', color = 'b')

    for i in range(len(expert_keys)):
        if i >= 4:
            offset = mtransforms.ScaledTranslation(0, 0.1 / 2.54, plt.gcf().dpi_scale_trans)
            transform = plt.gca().transData + offset
            plt.text(expert_keys[i], expert_values[i], f"{expert_values[i]:.2f}", ha='center', va='bottom', fontsize=8, transform=transform)
    
    # title     
    plt.plot(expert_keys, expert_values, marker='o', label='Single Expert', color = 'r')
    plt.title("[{}] Attention Layer and Single Eepert Latency in Decoding Phase".format(title))
    plt.xlabel("Batch Size")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.xticks(batch_sizes)
    
    plt.savefig("COMP_{}.png".format(title))



if __name__ == "__main__":
    title = "QWen"
    directory = 'measure_data_qwen'    
    main(directory, title)
