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
bigbird_times = [d['bigbird_time'] * 1000 for d in data]  
longformer_times = [d['longformer_time'] * 1000 for d in data]  
shadowy_times = [d['shadowy_time'] * 1000 for d in data]  
exposer_times = [d['exposer_time'] * 1000 for d in data]
dense_sparsity = [d['dense_sparsity'] for d in data]
bigbird_sparsity = [d['bigbird_sparsity'] for d in data]
longformer_sparsity = [d['longformer_sparsity'] for d in data]
shadowy_sparsity = [d['shadowy_sparsity'] for d in data]
exposer_sparsity = [d['exposer_sparsity'] for d in data]

# Calculate the speedup ratio for each layout
dense_speedup = []
bigbird_speedup = []
longformer_speedup = []
shadowy_speedup = []
exposer_speedup = []
for i in range(len(layers)):
    dense_speedup.append(dense_times[i] / dense_times[i])
    bigbird_speedup.append(dense_times[i] / bigbird_times[i])
    longformer_speedup.append(dense_times[i] / longformer_times[i])
    shadowy_speedup.append(dense_times[i] / shadowy_times[i])
    exposer_speedup.append(dense_times[i] / exposer_times[i])

# Number of groups  
num_layers = len(layers)  
  
# Setting up the bar width  
bar_width = 0.15 
  
# Setting the position of the bars on the x-axis  
r1 = np.arange(num_layers)  
r2 = [x + bar_width for x in r1]  
r3 = [x + bar_width for x in r2]  
r4 = [x + bar_width for x in r3]  
r5 = [x + bar_width for x in r4]

# Creating the figure
fig, ax1 = plt.subplots(figsize=(6.3, 1.5))
  
# Creating the bar plot
bar_colors = ['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#171A39']
ax1.bar(r1, dense_times, color=bar_colors[0], width=bar_width, edgecolor='black', label='Dense', linewidth=0.5)
ax1.bar(r2, shadowy_times, color=bar_colors[1], width=bar_width, edgecolor='black', label='Shadowy', linewidth=0.5)  
ax1.bar(r3, bigbird_times, color=bar_colors[2], width=bar_width, edgecolor='black', label='BigBird', linewidth=0.5)  
ax1.bar(r4, longformer_times, color=bar_colors[3], width=bar_width, edgecolor='black', label='Longformer', linewidth=0.5)  
ax1.bar(r5, exposer_times, color=bar_colors[4], width=bar_width, edgecolor='black', label='Exposer', linewidth=0.5)  

# Adding labels  
ax1.set_xlabel('Layer', fontsize=8)  
ax1.set_ylabel('Time (ms)', fontsize=8)
ax1.set_xticks([r + bar_width for r in range(num_layers)], layers)  
ax1.set_xticklabels([d['layer'] for d in data], fontsize=8)

# plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.yticks(fontsize=8)

# Creating legend & title for the bar plot  
# ax1.legend(bbox_to_anchor=(-0.1, 1.12, 0., .202), loc='lower left', ncol=5, mode="expand", borderaxespad=0., frameon=False, fontsize=10)

# Creating a secondary y-axis for the sparsity ratio line plot 
# line_colors = ['#958EA2', '#582156', '#385E88', '#AA2070', '#EC008C'] 
line_colors = ['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#171A39']
ax2 = ax1.twinx()
# ax2.plot(r1, dense_speedup, color=line_colors[0], marker='o', markersize=3, linestyle='--', linewidth=0.5, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(r2, shadowy_speedup, color=line_colors[1], marker='^', markersize=3, linestyle='--', linewidth=0.5, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(r3, bigbird_speedup, color=line_colors[2], marker='s', markersize=3, linestyle='--', linewidth=0.5, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(r4, longformer_speedup, color=line_colors[3], marker='d', markersize=3, linestyle='--', linewidth=0.5, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(r5, exposer_speedup, color=line_colors[4], marker='x', markersize=3, linestyle='--', linewidth=0.5, markeredgecolor='black', markeredgewidth=0.5)

# Adding labels for the secondary y-axis
ax2.set_ylabel('Speedup', fontsize=8)
plt.yticks(fontsize=8)

# Saving the figure
plt.tight_layout()
plt.savefig(args.output)
