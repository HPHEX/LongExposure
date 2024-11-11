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

# Colors
colors = ['#958EA2', '#582156', '#385E88', '#AA2070', '#EC008C']

# Constructing the DataFrame directly from the provided data  
# data = [
# {'sparsity': 0.0, 'dense': 0.002645135354995729, 'dss': 0.0029136895990371707, 'dsd': 0.0012435865640640258},
# {'sparsity': 0.1, 'dense': 0.002661683177947998, 'dss': 0.0025686835622787474, 'dsd': 0.0010659225511550907},
# {'sparsity': 0.2, 'dense': 0.0026769407844543453, 'dss': 0.0023227596855163574, 'dsd': 0.0009645465707778933},
# {'sparsity': 0.3, 'dense': 0.0026365542316436775, 'dss': 0.0020291174459457395, 'dsd': 0.0008476672041416172},
# {'sparsity': 0.4, 'dense': 0.0026926489210128786, 'dss': 0.001774837749004364, 'dsd': 0.0007380172860622407},
# {'sparsity': 0.5, 'dense': 0.002704383974075317, 'dss': 0.0015293235182762147, 'dsd': 0.0006531276798248292},
# {'sparsity': 0.6, 'dense': 0.0026364313602447514, 'dss': 0.0012370943975448607, 'dsd': 0.0005420851147174834},
# {'sparsity': 0.7, 'dense': 0.002701311979293823, 'dss': 0.0010015948796272276, 'dsd': 0.0004518092793226243},
# {'sparsity': 0.8, 'dense': 0.002628689913749695, 'dss': 0.0007370342433452606, 'dsd': 0.0003428147190809248},
# {'sparsity': 0.9, 'dense': 0.0026712268543243422, 'dss': 0.0005301862406730651, 'dsd': 0.0002521702367067337}
# ]

# Extracting layer labels and corresponding times for each layout
sparsity_ratios = [d['sparsity'] for d in data]
dense_time = [d['dense'] for d in data]
dss_time = [d['dss'] for d in data]
dsd_time = [d['dsd'] * 2 for d in data]

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
plt.plot(r1, dense_time, color=line_colors[0], marker='o', label='Dense', markersize=3, linewidth=1)
plt.plot(r2, dss_time, color=line_colors[1], marker='^', label='DSS', markersize=3, linewidth=1)
plt.plot(r3, dsd_time, color=line_colors[2], marker='s', label='DSD', markersize=3, linewidth=1)
  
# Adding labels  
plt.xlabel('Sparsity Ratio', fontweight='bold')  
plt.ylabel('Execution Time', fontweight='bold')
plt.xticks([r + bar_width for r in range(num_ratios)], sparsity_ratios)
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 

# Creating legend & title for the bar plot  
plt.legend(bbox_to_anchor=(0.2, 1.02, 1.0, 1.02), loc='lower left', ncol=3, mode="expand", borderaxespad=0., frameon=False, fontsize=8)

# Saving the figure (optional)  
plt.tight_layout()
plt.savefig(args.output)
