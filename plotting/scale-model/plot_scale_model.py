import matplotlib.pyplot as plt
import json
import argparse

parser = argparse.ArgumentParser(description='Attn block sparse')
parser.add_argument('--save_json', type=str, default=None, help='save')
parser.add_argument('--output', type=str, default=None, help='output')
args = parser.parse_args()


data = [
    # A100 large 512
    {"task": "a100-large-512-4-Full Parameter",          "time": 0.0},
    {"task": "a100-large-512-4-Lora",                    "time": 0.0},
    {"task": "a100-large-512-4-Long Exposure + Lora",    "time": 0.0},
    {"task": "a100-large-512-4-Adapter",                 "time": 0.0},
    {"task": "a100-large-512-4-Long Exposure + Adapter", "time": 0.0},
    {"task": "a100-large-512-4-Bitfit",                  "time": 0.0},
    {"task": "a100-large-512-4-Long Exposure + Bitfit",  "time": 0.0},
    # A100 large 1024
    {"task": "a100-large-1024-4-Full Parameter",          "time": 0.0},
    {"task": "a100-large-1024-4-Lora",                    "time": 0.0},
    {"task": "a100-large-1024-4-Long Exposure + Lora",    "time": 0.0},
    {"task": "a100-large-1024-4-Adapter",                 "time": 0.0},
    {"task": "a100-large-1024-4-Long Exposure + Adapter", "time": 0.0},
    {"task": "a100-large-1024-4-Bitfit",                  "time": 0.0},
    {"task": "a100-large-1024-4-Long Exposure + Bitfit",  "time": 0.0},
    # A100 xl 512
    {"task": "a100-xl-512-4-Full Parameter",          "time": 0.0},
    {"task": "a100-xl-512-4-Lora",                    "time": 0.0},
    {"task": "a100-xl-512-4-Long Exposure + Lora",    "time": 0.0},
    {"task": "a100-xl-512-4-Adapter",                 "time": 0.0},
    {"task": "a100-xl-512-4-Long Exposure + Adapter", "time": 0.0},
    {"task": "a100-xl-512-4-Bitfit",                  "time": 0.0},
    {"task": "a100-xl-512-4-Long Exposure + Bitfit",  "time": 0.0},
    # A100 xl 1024
    {"task": "a100-xl-1024-4-Full Parameter",          "time": 0.0},
    {"task": "a100-xl-1024-4-Lora",                    "time": 0.0},
    {"task": "a100-xl-1024-4-Long Exposure + Lora",    "time": 0.0},
    {"task": "a100-xl-1024-4-Adapter",                 "time": 0.0},
    {"task": "a100-xl-1024-4-Long Exposure + Adapter", "time": 0.0},
    {"task": "a100-xl-1024-4-Bitfit",                  "time": 0.0},
    {"task": "a100-xl-1024-4-Long Exposure + Bitfit",  "time": 0.0},
]

with open(args.save_json, 'r') as f:
    real_data = json.load(f)

for record in data:
    for dataa in real_data:
        if record['task'] == dataa['task']:
            record['time'] = dataa['time']
# data = [
#     # A100 GPT2-large 512
#     # {"task": "a100-large-512-4-Full Parameter", "time": 0.0},
# ]


tasks = [d['task'] for d in data]
times = [d['time'] for d in data]

default_width = 0.8
gap_width = 0.2

plt.figure(figsize=(8, 1.5))
colors=['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3']

# A100 large 512
plt.bar(1 - 0.1, times[0], color=colors[0], width=default_width, edgecolor='black')

plt.bar(2 + 0.1, times[1], color=colors[1], width=default_width, edgecolor='black')
plt.bar(3 - 0.1, times[2], color=colors[1], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[1] / times[2])

plt.bar(4 + 0.1, times[3], color=colors[2], width=default_width, edgecolor='black')
plt.bar(5 - 0.1, times[4], color=colors[2], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[3] / times[4])

plt.bar(6 + 0.1, times[5], color=colors[3], width=default_width, edgecolor='black')
plt.bar(7 - 0.1, times[6], color=colors[3], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[5] / times[6])

# A100 large 1024
plt.bar(9 - 0.1, times[7], color=colors[0], width=default_width, edgecolor='black')

plt.bar(10 + 0.1, times[8], color=colors[1], width=default_width, edgecolor='black')
plt.bar(11 - 0.1, times[9], color=colors[1], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[8] / times[9])

plt.bar(12 + 0.1, times[10], color=colors[2], width=default_width, edgecolor='black')
plt.bar(13 - 0.1, times[11], color=colors[2], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[10] / times[11])

plt.bar(14 + 0.1, times[12], color=colors[3], width=default_width, edgecolor='black')
plt.bar(15 - 0.1, times[13], color=colors[3], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[12] / times[13])

# A100 xl 512
plt.bar(17 - 0.1, times[14], color=colors[0], width=default_width, edgecolor='black')

plt.bar(18 + 0.1, times[15], color=colors[1], width=default_width, edgecolor='black')
plt.bar(19 - 0.1, times[16], color=colors[1], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[15] / times[16])

plt.bar(20 + 0.1, times[17], color=colors[2], width=default_width, edgecolor='black')
plt.bar(21 - 0.1, times[18], color=colors[2], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[17] / times[18])

plt.bar(22 + 0.1, times[19], color=colors[3], width=default_width, edgecolor='black')
plt.bar(23 - 0.1, times[20], color=colors[3], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[19] / times[20])

# A100 xl 1024
plt.bar(25 - 0.1, times[21], color=colors[0], width=default_width, edgecolor='black')

plt.bar(26 + 0.1, times[22], color=colors[1], width=default_width, edgecolor='black')
plt.bar(27 - 0.1, times[23], color=colors[1], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[22] / times[23])

plt.bar(28 + 0.1, times[24], color=colors[2], width=default_width, edgecolor='black')
plt.bar(29 - 0.1, times[25], color=colors[2], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[24] / times[25])

plt.bar(30 + 0.1, times[26], color=colors[3], width=default_width, edgecolor='black')
plt.bar(31 - 0.1, times[27], color=colors[3], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[26] / times[27])

plt.ylabel('Time (ms)', fontsize=12)
plt.gca().axes.get_xaxis().set_visible(False)
plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.legend(['Full Parameter', 'LoRA', 'Long Exposure + LoRA', 'Adapter', 'Long Exposure + Adapter', 'BitFit', 'Long Exposure + BitFit'], loc='upper left', fontsize=10, frameon=False)

plt.tight_layout()
plt.savefig(args.output)
