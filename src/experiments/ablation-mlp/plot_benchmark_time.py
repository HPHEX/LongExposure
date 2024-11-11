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
  
# data = [
# {'layer': 0, 'dense_time': 0.0042661683177948, 'column_sparse_time': 0.0008999116766452791, 'block_sparse_time': 0.0008529510390758514},
# {'layer': 1, 'dense_time': 0.004296642541885376, 'column_sparse_time': 0.002567167978286743, 'block_sparse_time': 0.0008281497573852538},
# {'layer': 2, 'dense_time': 0.004305367012023925, 'column_sparse_time': 0.0037673983860015857, 'block_sparse_time': 0.0011463884782791135},
# {'layer': 3, 'dense_time': 0.004329021425247192, 'column_sparse_time': 0.0008597504031658174, 'block_sparse_time': 0.0008707276809215546},
# {'layer': 4, 'dense_time': 0.004295229396820069, 'column_sparse_time': 0.0010614374327659606, 'block_sparse_time': 0.0008905727994441985},
# {'layer': 5, 'dense_time': 0.004304752616882324, 'column_sparse_time': 0.0016641228818893442, 'block_sparse_time': 0.0008101478397846223},
# {'layer': 6, 'dense_time': 0.004302663660049437, 'column_sparse_time': 0.0011373158454895023, 'block_sparse_time': 0.0009512140834331515},
# {'layer': 7, 'dense_time': 0.004294144020080566, 'column_sparse_time': 0.0017825996828079228, 'block_sparse_time': 0.0008816844797134402},
# {'layer': 8, 'dense_time': 0.004317163496017456, 'column_sparse_time': 0.0011192729544639588, 'block_sparse_time': 0.0009848012864589687},
# {'layer': 9, 'dense_time': 0.0043163647985458384, 'column_sparse_time': 0.001007349752187729, 'block_sparse_time': 0.0009417113602161406},
# {'layer': 10, 'dense_time': 0.004296396799087524, 'column_sparse_time': 0.001132564468383789, 'block_sparse_time': 0.0009330278444290162},
# {'layer': 11, 'dense_time': 0.004318064661026001, 'column_sparse_time': 0.0011160780739784242, 'block_sparse_time': 0.0009094348800182345},
# {'layer': 12, 'dense_time': 0.004322181100845338, 'column_sparse_time': 0.0013389824151992802, 'block_sparse_time': 0.000987422728538513},
# {'layer': 13, 'dense_time': 0.0043171225070953384, 'column_sparse_time': 0.0009934847867488859, 'block_sparse_time': 0.000911298565864563},
# {'layer': 14, 'dense_time': 0.004327116775512696, 'column_sparse_time': 0.0011179007959365847, 'block_sparse_time': 0.0009313280022144317},
# {'layer': 15, 'dense_time': 0.004304261140823365, 'column_sparse_time': 0.001152511990070343, 'block_sparse_time': 0.0009337036788463596},
# {'layer': 16, 'dense_time': 0.0043291648292541505, 'column_sparse_time': 0.0009999155211448676, 'block_sparse_time': 0.0010403225636482238},
# {'layer': 17, 'dense_time': 0.004325068845748902, 'column_sparse_time': 0.0012141158199310302, 'block_sparse_time': 0.0009723084807395934},
# {'layer': 18, 'dense_time': 0.004316200952529907, 'column_sparse_time': 0.0011238809800148011, 'block_sparse_time': 0.0010305740857124328},
# {'layer': 19, 'dense_time': 0.004332605466842651, 'column_sparse_time': 0.0011648614430427552, 'block_sparse_time': 0.0009928294467926027},
# {'layer': 20, 'dense_time': 0.004323143701553344, 'column_sparse_time': 0.0011456307101249696, 'block_sparse_time': 0.0010240409636497498},
# {'layer': 21, 'dense_time': 0.0043245568084716805, 'column_sparse_time': 0.0012905267214775086, 'block_sparse_time': 0.000976875523328781},
# {'layer': 22, 'dense_time': 0.004318720016479491, 'column_sparse_time': 0.001179217920303345, 'block_sparse_time': 0.0009208627283573154},
# {'layer': 23, 'dense_time': 0.004327608308792116, 'column_sparse_time': 0.001129369604587555, 'block_sparse_time': 0.000976302078962326},
# ]

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
