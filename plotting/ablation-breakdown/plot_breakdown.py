import matplotlib.pyplot as plt  
import json
import argparse

parser = argparse.ArgumentParser(description='Attn block sparse')
parser.add_argument('--save_json', type=str, default=None, help='save')
parser.add_argument('--output', type=str, default=None, help='save')
args = parser.parse_args()
  
data = [
    {"test_case": "Long Exposure+Bitfit",  "total_time": 0.0, "forward": 0.0, "backward": 0.0, "optimizer step": 0.0, "prediction": 0.0},
    {"test_case": "BitFit",                "total_time": 0.0, "forward": 0.0, "backward": 0.0, "optimizer step": 0.0, "prediction": 0.0},
    {"test_case": "Long Exposure+Adapter", "total_time": 0.0, "forward": 0.0, "backward": 0.0, "optimizer step": 0.0, "prediction": 0.0},
    {"test_case": "Adapter",               "total_time": 0.0, "forward": 0.0, "backward": 0.0, "optimizer step": 0.0, "prediction": 0.0},
    {"test_case": "Long Exposure+LoRA",    "total_time": 0.0, "forward": 0.0, "backward": 0.0, "optimizer step": 0.0, "prediction": 0.0},
    {"test_case": "LoRA",                  "total_time": 0.0, "forward": 0.0, "backward": 0.0, "optimizer step": 0.0, "prediction": 0.0},
    {"test_case": "Full Parameter",        "total_time": 0.0, "forward": 0.0, "backward": 0.0, "optimizer step": 0.0, "prediction": 0.0},
]
with open(args.save_json, 'r') as f:
    new_data = json.load(f)

for record in data:
    for new_record in new_data:
        if record['test_case'] == new_record['test_case']:
            record['total_time'] = new_record['total_time']
            record['forward'] = new_record['fwd_time']
            record['backward'] = new_record['bwd_time']
            record['optimizer step'] = new_record['opt_time']
            record['prediction'] = new_record['predict_time']

# Names of the test cases  
test_cases = [d['test_case'] for d in data]  
  
# The different stages we want to plot  
# colors = ['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#171A39', '#FF7F11', '#FFC0A9', '#FFB7B2', '#FFD8D0', '#FFC8B2']
stages = ['forward', 'backward', 'optimizer step', 'prediction']
stage_colors = {'forward': '#0C408C', 'backward': '#8186D8', 'optimizer step': '#BF84BA', 'prediction': '#FFDFD3'}

# Create a figure and a set of subplots  
fig, ax = plt.subplots(figsize=(8, 2.5))

# Set the bar height
bar_height = 0.1
bar_y_based = 0.5
bar_y_gap = 0.1

# Loop over the data to create stacked horizontal bars  
for i, d in enumerate(data):  
    left = 0  # Starting point for the first stage  
    y_pos = i * (bar_height + bar_y_gap) + bar_y_based
    for stage in stages:  
        # Plot each stage in the bar  
        ax.barh(y_pos, d[stage], left=left, label=stage if i == 0 else "", color=stage_colors[stage], height=bar_height, edgecolor='black')
        left += d[stage]  # Update the starting point for the next stage  
  
# Add labels and title  
ax.set_xlabel('Execution Time (ms)', fontsize=12)
# ax.set_title('OPT-1.3B Fine-tuning Performance Breakdown')
ax.set_yticks([i * (bar_height + bar_y_gap) + bar_y_based for i in range(len(data))])
ax.set_yticklabels(test_cases, fontsize=12)
ax.set_xlim(0, 1.1 * max([d['total_time'] for d in data]))
  
# Add a legend
ax.legend(loc='lower right', bbox_to_anchor=(1, 0), frameon=False)

# Save the figure
plt.tight_layout()
plt.savefig(args.output, bbox_inches='tight')
