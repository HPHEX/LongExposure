import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

parser = argparse.ArgumentParser(description='Attn block sparse')
parser.add_argument('save_json', type=str, help='save json file')
parser.add_argument('output', type=str, help='output file')

args = parser.parse_args()
with open(args.save_json, 'r') as f:
    data_real = json.load(f) 

data = [
    # OPT-1.3b batch_size 2 seq_len 1024
    {'task': 'exposer + lora',    'num_cards': 1, 'time': 0.0},
    {'task': 'exposer + lora',    'num_cards': 2, 'time': 0.0},
    {'task': 'exposer + lora',    'num_cards': 4, 'time': 0.0},
    {'task': 'exposer + adapter', 'num_cards': 1, 'time': 0.0},
    {'task': 'exposer + adapter', 'num_cards': 2, 'time': 0.0},
    {'task': 'exposer + adapter', 'num_cards': 4, 'time': 0.0},
    {'task': 'exposer + bitfit',  'num_cards': 1, 'time': 0.0},
    {'task': 'exposer + bitfit',  'num_cards': 2, 'time': 0.0},
    {'task': 'exposer + bitfit',  'num_cards': 4, 'time': 0.0},
]

for record in data:
    for new_re in data_real:
        if new_re['task'] == record['task'] and new_re['num_cards'] == record['num_cards']:
            record['time'] = new_re['time']
            break


# Extracting layer labels and corresponding times for each layout
time_exposer_lora = [1000 * 1024 / d['time'] for d in data if d['task'] == 'exposer + lora']
time_exposer_adapter = [1000 * 1024 / d['time'] for d in data if d['task'] == 'exposer + adapter']
time_exposer_bitfit = [1000 * 1024 / d['time'] for d in data if d['task'] == 'exposer + bitfit']
  
# Setting the position of the bars on the x-axis  
r0 = np.arange(3)

# Creating the figure
plt.figure(figsize=(4, 3))
  
# Creating the bar plot
line_colors = ['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#171A39', '#FFA07A']
plt.plot(r0, time_exposer_lora, color=line_colors[2], marker='o', label='Exposer + LoRA', markersize=7, linewidth=2)
plt.plot(r0, time_exposer_adapter, color=line_colors[4], marker='^', label='Exposer + Adapter', markersize=7, linewidth=2)
plt.plot(r0, time_exposer_bitfit, color=line_colors[5], marker='s', label='Exposer + BitFit', markersize=7, linewidth=2)
  
# Adding labels  
plt.xticks([r for r in range(3)], ['1', '2', '4'])

# Adding legend
plt.legend(loc='upper left', ncol=1, fontsize=8, frameon=False)

# Saving the figure (optional)  
plt.tight_layout()
plt.savefig(args.output)
