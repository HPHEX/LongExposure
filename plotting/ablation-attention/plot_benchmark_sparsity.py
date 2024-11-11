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
# dense_sparsity = [1.0 - d['dense_sparsity'] for d in data]
bigbird_sparsity = [1.0 - d['bigbird_sparsity'] for d in data]
longformer_sparsity = [1.0 - d['longformer_sparsity'] for d in data]
shadowy_sparsity = [1.0 - d['shadowy_sparsity'] for d in data]
exposer_sparsity = [1.0 - d['exposer_sparsity'] for d in data]

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
plt.figure(figsize=(4, 3))
  
# Creating the bar plot
line_colors = ['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#171A39']
# plt.plot(r1, dense_sparsity, color=line_colors[0], marker='o', label='Dense', markersize=3, linewidth=1)
plt.plot(r2, shadowy_sparsity, color=line_colors[0], marker='^', label='Shadowy', markersize=3, linewidth=1)
plt.plot(r3, bigbird_sparsity, color=line_colors[1], marker='s', label='BigBird', markersize=3, linewidth=1)
plt.plot(r4, longformer_sparsity, color=line_colors[2], marker='d', label='Longformer', markersize=3, linewidth=1)
plt.plot(r5, exposer_sparsity, color=line_colors[4], marker='x', label='Long Exposure', markersize=3, linewidth=1)
  
# Adding labels  
plt.xlabel('Layer') 
plt.ylabel('Sparsity Ratio')
plt.xticks([r + bar_width for r in range(num_layers)], layers, fontsize=10, rotation=90)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=10)

# Creating legend & title for the bar plot  
# plt.legend(bbox_to_anchor=(-0.2, 1.02, 1.2, 1.02), loc='lower left', ncol=5, mode="expand", borderaxespad=0., frameon=False, fontsize=8)
plt.legend(loc='lower right', fontsize=10, frameon=False)
plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.5)

# Saving the figure (optional)  
plt.tight_layout()
plt.savefig(args.output)
