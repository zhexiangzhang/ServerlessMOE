with open('data.txt', 'r') as file:
    lines = file.readlines()

att_values = []
moe_values = []

for line in lines:    
    att_index = line.find('att')
    moe_index = line.find('moe')
    att_value = float(line[att_index:].split('=')[1].split(',')[0].strip())
    moe_value = float(line[moe_index:].split('=')[1].strip())
    
    att_values.append(att_value)
    moe_values.append(moe_value)

average_att = sum(att_values) / len(att_values)
average_moe = sum(moe_values) / len(moe_values)
max_att = max(att_values)
min_att = min(att_values)
max_moe = max(moe_values)
min_moe = min(moe_values)

print("Average att:", round(average_att, 5))
print("Average moe:", round(average_moe, 5))
print("Max att:", round(max_att, 5), "Min att:", round(min_att, 5))
print("Max moe:", round(max_moe, 5), "Min moe:", round(min_moe, 5))

