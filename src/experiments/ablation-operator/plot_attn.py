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

# Constructing the DataFrame directly from the provided data  
# data = [
# {'sparsity': 0.0, 'dense_sdd': 0.0024245657682418825, 'sdd': 0.002386001930236816, 'dense_dsd': 0.0014777958393096921, 'dsd': 0.0017894399881362912, 'torch_sdd': 0.0, 'torch_dsd': 0.0},
# {'sparsity': 0.1, 'dense_sdd': 0.002432081894874573, 'sdd': 0.00219721727848053, 'dense_dsd': 0.0014784921574592586, 'dsd': 0.0016678092789649962, 'torch_sdd': 0.0, 'torch_dsd': 0.0},
# {'sparsity': 0.2, 'dense_sdd': 0.002473697261810303, 'sdd': 0.001985269768238068, 'dense_dsd': 0.001479843831062317, 'dsd': 0.0014798643255233765, 'torch_sdd': 0.0, 'torch_dsd': 0.0},
# {'sparsity': 0.3, 'dense_sdd': 0.002465484805107117, 'sdd': 0.0017419263982772827, 'dense_dsd': 0.0014786764860153197, 'dsd': 0.0013025689601898192, 'torch_sdd': 0.0, 'torch_dsd': 0.0},
# {'sparsity': 0.4, 'dense_sdd': 0.0024449843168258666, 'sdd': 0.0015380889630317687, 'dense_dsd': 0.0014792294406890869, 'dsd': 0.0011461632037162782, 'torch_sdd': 0.0, 'torch_dsd': 0.0},
# {'sparsity': 0.5, 'dense_sdd': 0.0024361164760589604, 'sdd': 0.0012848742246627807, 'dense_dsd': 0.0014804377579689023, 'dsd': 0.0009588735997676854, 'torch_sdd': 0.0, 'torch_dsd': 0.0},
# {'sparsity': 0.6, 'dense_sdd': 0.002433638410568237, 'sdd': 0.0010792140769958494, 'dense_dsd': 0.0014809702348709106, 'dsd': 0.0008060723197460177, 'torch_sdd': 0.0, 'torch_dsd': 0.0},
# {'sparsity': 0.7, 'dense_sdd': 0.002450022411346435, 'sdd': 0.0008321024024486541, 'dense_dsd': 0.001479966716766357, 'dsd': 0.0006259711980819703, 'torch_sdd': 0.0, 'torch_dsd': 0.0},
# {'sparsity': 0.8, 'dense_sdd': 0.0024755609703063968, 'sdd': 0.0006131916773319246, 'dense_dsd': 0.0014789836788177484, 'dsd': 0.00047124479889869677, 'torch_sdd': 0.0, 'torch_dsd': 0.0},
# {'sparsity': 0.9, 'dense_sdd': 0.0024559206485748287, 'sdd': 0.00038893568158149716, 'dense_dsd': 0.0014804582357406614, 'dsd': 0.00035301375865936293, 'torch_sdd': 0.0, 'torch_dsd': 0.0}
# ]

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
