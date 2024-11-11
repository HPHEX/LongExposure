import argparse
import matplotlib.pyplot as plt    
import numpy as np
import json

parser = argparse.ArgumentParser(description='Attn block sparse')
parser.add_argument('--save_json', type=str, default=None, help='save')
parser.add_argument('--save_png', type=str, default=None, help='save')

args = parser.parse_args()

with open(args.save_json, 'r') as f:
    data = json.load(f)
# data = [    
#     {"Seq_len": 128,  "Full Parameter": 6.847414493560791, "LoRA": 2.8141188621520996, "Long Exposure+LoRA": 3.12906551361084, "Long Exposure+LoRA (optimal)": 2.892338752746582},    
#     {"Seq_len": 256,  "Full Parameter": 6.894482135772705, "LoRA": 4.1683783531188965, "Long Exposure+LoRA": 3.9481968879699707, "Long Exposure+LoRA (optimal)": 3.6681666374206543},    
#     {"Seq_len": 512,  "Full Parameter": 11.024869441986084, "LoRA": 8.259130001068115, "Long Exposure+LoRA": 5.555179595947266, "Long Exposure+LoRA (optimal)": 5.179873466491699},    
#     {"Seq_len": 1024, "Full Parameter": 25.220609188079834, "LoRA": 22.068013668060303, "Long Exposure+LoRA": 8.7814040184021, "Long Exposure+LoRA (optimal)": 8.214815616607666},    
#     {"Seq_len": 2048, "Full Parameter": 76.11208868026733, "LoRA": 72.1869101524353, "Long Exposure+LoRA": 15.216221332550049, "Long Exposure+LoRA (optimal)": 14.27677059173584},    
# ]      
    
colors = ['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#171A39']    
    
# Extracting data    
seq_lens = [d["Seq_len"] for d in data]    
full_param = [d["Full Parameter"] for d in data]    
lora = [d["LoRA"] for d in data]    
long_exposure_lora = [d["Long Exposure+LoRA"] for d in data]    
long_exposure_lora_optimal = [d["Long Exposure+LoRA (optimal)"] for d in data]    
    
# Number of groups and bar width    
num_groups = len(seq_lens)    
bar_width = 0.18
    
# Setting the position of the bars on the x-axis    
r1 = np.arange(num_groups)    
r2 = [x + bar_width for x in r1]    
r3 = [x + bar_width for x in r2]    
r4 = [x + bar_width for x in r3]    
    
# Create the figure and the bar plot    
plt.figure(figsize=(4, 3))  
    
plt.bar(r1, full_param, color=colors[0], width=bar_width, label='Full Parameter', edgecolor='black')   
plt.bar(r2, lora, color=colors[1], width=bar_width, label='LoRA', edgecolor='black')    
plt.bar(r3, long_exposure_lora, color=colors[2], width=bar_width, label='Long Exposure', edgecolor='black')    
plt.bar(r4, long_exposure_lora_optimal, color=colors[3], width=bar_width, label='Long Exposure(optimal)', edgecolor='black')    
    
# Adding labels    
# plt.xlabel('Sequence Length')    
plt.ylabel('GPU Memory (GB)', fontsize=12)
plt.yticks(fontsize=12)
plt.xticks([r + bar_width for r in range(num_groups)], seq_lens, fontsize=12)   
    
# Creating legend & title for the bar plot    
# plt.legend(bbox_to_anchor=(-0.12, 1.2, 1.15, .202), loc='lower left', ncol=2, mode="expand", borderaxespad=0., frameon=False, fontsize=12)  

# Adding grid
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Saving the figure  
plt.tight_layout()  
plt.savefig(args.save_png) 