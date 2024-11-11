import matplotlib.pyplot as plt
import json
import argparse  
import numpy as np 
  
parser = argparse.ArgumentParser()
parser.add_argument('--save_json', type=str, default='None', help='save')
parser.add_argument('--output', type=str, default='None', help='output file')
args = parser.parse_args()
with open(args.save_json, 'r') as f:
    data = json.load(f)

# Extracting layer labels and corresponding times for each layout  
layers = [d['layer'] for d in data]  
dense_times = [d['dense_time'] * 1000 for d in data]  
column_sparse_times = [d['column_sparse_time'] * 1000 for d in data]   
block_sparse_times = [d['block_sparse_time'] * 1000 for d in data]

# Calculate the speedup ratio for each layout
dense_speedup = []
column_sparse_speedup = []
block_sparse_speedup = []
for i in range(len(layers)):
    dense_speedup.append(dense_times[i] / dense_times[i])
    column_sparse_speedup.append(dense_times[i] / column_sparse_times[i])
    block_sparse_speedup.append(dense_times[i] / block_sparse_times[i])
  
# Number of groups  
num_layers = len(layers)  

# Setting up the bar width  
bar_width = 0.25
  
# Setting the position of the bars on the x-axis  
r1 = np.arange(num_layers)  
r2 = [x + bar_width for x in r1]  
r3 = [x + bar_width for x in r2]  
r4 = [x + bar_width for x in r3]  
r5 = [x + bar_width for x in r4]

# Creating the figure
fig, ax1 = plt.subplots(figsize=(6, 1.5))
  
# Creating the bar plot
bar_colors = ['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#171A39']
ax1.bar(r1, dense_times, color=bar_colors[0], width=bar_width, label='Dense', edgecolor='black', linewidth=0.5)
ax1.bar(r2, column_sparse_times, color=bar_colors[1], width=bar_width, label='Shadowy', edgecolor='black', linewidth=0.5)
ax1.bar(r3, block_sparse_times, color=bar_colors[2], width=bar_width, label='Long Exposure', edgecolor='black', linewidth=0.5)
  
# Adding labels  
ax1.set_xlabel('Layer', fontsize=8)  
ax1.set_ylabel('Time (ms)', fontsize=8)
ax1.set_xticks([r + bar_width for r in range(num_layers)], layers)  
ax1.set_xticklabels([d['layer'] for d in data])
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Creating legend & title for the bar plot  
ax1.legend(bbox_to_anchor=(-0.05, 1.12, 1.1, .202), loc='lower left', ncol=5, mode="expand", borderaxespad=0., frameon=False, fontsize=10)
  
# Creating a secondary y-axis for the sparsity ratio line plot 
# line_colors = ['#958EA2', '#582156', '#385E88', '#AA2070', '#EC008C'] 
line_colors = ['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#171A39']
ax2 = ax1.twinx()
# ax2.plot(r1, dense_speedup, color=line_colors[0], marker='o', markersize=3, linestyle='--', linewidth=0.5, markeredgecolor='black')
ax2.plot(r2, column_sparse_speedup, color=line_colors[1], marker='o', markersize=3, linestyle='--', linewidth=0.5, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(r3, block_sparse_speedup, color=line_colors[2], marker='o', markersize=3, linestyle='--', linewidth=0.5, markeredgecolor='black', markeredgewidth=0.5)

# Adding labels for the secondary y-axis
ax2.set_ylabel('Speedup', fontsize=8)
plt.yticks(fontsize=8)

# Saving the figure (optional)  
plt.tight_layout()
plt.savefig(args.output)
