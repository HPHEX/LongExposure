import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

parser = argparse.ArgumentParser()
parser.add_argument('--save_json', type=str, default='None', help='save')
parser.add_argument('--output', type=str, default='None', help='output file')
args = parser.parse_args()
# Colors
colors = ['#958EA2', '#582156', '#385E88', '#AA2070', '#EC008C']

with open(args.save_json, 'r') as f:
    data = json.load(f)

# Extracting layer labels and corresponding times for each layout
sparsity_ratios = [d['sparsity'] for d in data]
dense_sdd_sparsity = [d['dense_sdd'] for d in data]
sdd_sparsity = [d['sdd'] for d in data]
dense_dsd_sparsity = [d['dense_dsd'] for d in data]
dsd_sparsity = [d['dsd'] for d in data]
  
# Number of groups  
num_ratios = len(sparsity_ratios)
  
# Setting up the bar width  
bar_width = 0.15 
  
# Setting the position of the bars on the x-axis  
r1 = np.arange(num_ratios)
r2 = [x + bar_width for x in r1]  
r3 = [x + bar_width for x in r1]  
r4 = [x + bar_width for x in r1]  
r5 = [x + bar_width for x in r1]

# Creating the figure
plt.figure(figsize=(4, 3))
  
# Creating the bar plot
line_colors = ['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#171A39']
plt.plot(r1, dense_sdd_sparsity, color=line_colors[0], marker='o', label='Dense SDD', markersize=3, linewidth=1)
plt.plot(r2, sdd_sparsity, color=line_colors[1], marker='^', label='SDD', markersize=3, linewidth=1)
plt.plot(r3, dense_dsd_sparsity, color=line_colors[2], marker='s', label='Dense DSD', markersize=3, linewidth=1)
plt.plot(r4, dsd_sparsity, color=line_colors[3], marker='d', label='DSD', markersize=3, linewidth=1)

  
# Adding labels  
plt.xlabel('Sparsity Ratio', fontweight='bold')  
plt.ylabel('Execution Time', fontweight='bold')
plt.xticks([r + bar_width for r in range(num_ratios)], sparsity_ratios)
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 

# Creating legend & title for the bar plot  
plt.legend(bbox_to_anchor=(0.2, 1.02, 0.8, 1.02), loc='lower left', ncol=2, mode="expand", borderaxespad=0., frameon=False, fontsize=8)

# Saving the figure (optional)  
plt.tight_layout()
plt.savefig(args.output)
