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
# {'layer': 0, 'dense_time': 0.013744988260269167, 'bigbird_time': 0.004891340818405151, 'longformer_time': 0.00506793984413147, 'shadowy_time': 0.005910609912872313, 'exposer_time': 0.004643860483169556, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.2568359375, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.0052490234375},
# {'layer': 1, 'dense_time': 0.007904296970367429, 'bigbird_time': 0.004766699542999269, 'longformer_time': 0.005030789136886596, 'shadowy_time': 0.005931171808242799, 'exposer_time': 0.004379668464660646, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.2607421875, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.005035400390625},
# {'layer': 2, 'dense_time': 0.00790730751991272, 'bigbird_time': 0.004751073303222657, 'longformer_time': 0.0050230476856231695, 'shadowy_time': 0.005916057624816894, 'exposer_time': 0.004405719022750855, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.2578125, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.005279541015625},
# {'layer': 3, 'dense_time': 0.007867658243179322, 'bigbird_time': 0.0046308761882781985, 'longformer_time': 0.0049451417636871345, 'shadowy_time': 0.005834035196304318, 'exposer_time': 0.004124262371063232, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.255859375, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.005096435546875},
# {'layer': 4, 'dense_time': 0.007846133794784547, 'bigbird_time': 0.004685967359542848, 'longformer_time': 0.00498710527420044, 'shadowy_time': 0.0058636082935333254, 'exposer_time': 0.004020244469642639, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.26171875, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.00518798828125},
# {'layer': 5, 'dense_time': 0.007840808973312382, 'bigbird_time': 0.004637102098464966, 'longformer_time': 0.004941189126968383, 'shadowy_time': 0.005840322542190552, 'exposer_time': 0.004075724782943726, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.255859375, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.005126953125},
# {'layer': 6, 'dense_time': 0.00785086463928223, 'bigbird_time': 0.004665487346649171, 'longformer_time': 0.00494090238571167, 'shadowy_time': 0.005841264629364014, 'exposer_time': 0.004178001914024354, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.2607421875, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.00518798828125},
# {'layer': 7, 'dense_time': 0.007884185609817504, 'bigbird_time': 0.004660142059326172, 'longformer_time': 0.004977029123306273, 'shadowy_time': 0.005895004158020019, 'exposer_time': 0.00422436863422394, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.2548828125, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.00518798828125},
# {'layer': 8, 'dense_time': 0.007844044799804687, 'bigbird_time': 0.004651950092315674, 'longformer_time': 0.004956712913513184, 'shadowy_time': 0.005874974737167357, 'exposer_time': 0.004010250253677367, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.2548828125, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.00213623046875},
# {'layer': 9, 'dense_time': 0.00785115135192871, 'bigbird_time': 0.0046619648170471185, 'longformer_time': 0.004918743057250976, 'shadowy_time': 0.005870366716384889, 'exposer_time': 0.004000440325737002, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.2568359375, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.0047607421875},
# {'layer': 10, 'dense_time': 0.007865077791213987, 'bigbird_time': 0.004648632326126099, 'longformer_time': 0.00496218111038208, 'shadowy_time': 0.005847306241989137, 'exposer_time': 0.004195061769485474, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.2548828125, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.001983642578125},
# {'layer': 11, 'dense_time': 0.007898705911636354, 'bigbird_time': 0.004653629426956176, 'longformer_time': 0.004927999992370606, 'shadowy_time': 0.005816012802124023, 'exposer_time': 0.004086476812362672, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.2529296875, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.001922607421875},
# {'layer': 12, 'dense_time': 0.007906017274856568, 'bigbird_time': 0.004666429452896117, 'longformer_time': 0.004956999683380127, 'shadowy_time': 0.005904855031967165, 'exposer_time': 0.0042728652667999275, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.255859375, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.002105712890625},
# {'layer': 13, 'dense_time': 0.007909949417114256, 'bigbird_time': 0.004667535362243652, 'longformer_time': 0.0049811251068115235, 'shadowy_time': 0.005889802236557007, 'exposer_time': 0.004241469469070435, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.2578125, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.001983642578125},
# {'layer': 14, 'dense_time': 0.00792012797355652, 'bigbird_time': 0.004687667198181153, 'longformer_time': 0.004995870733261107, 'shadowy_time': 0.005885726680755613, 'exposer_time': 0.004046438393592834, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.2548828125, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.00213623046875},
# {'layer': 15, 'dense_time': 0.007892684755325315, 'bigbird_time': 0.004678676490783692, 'longformer_time': 0.004957143030166626, 'shadowy_time': 0.005861375970840454, 'exposer_time': 0.004086579208374024, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25390625, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.0020751953125},
# {'layer': 16, 'dense_time': 0.007867801628112795, 'bigbird_time': 0.0046720818901061995, 'longformer_time': 0.00493981698989868, 'shadowy_time': 0.00589041666030884, 'exposer_time': 0.0040593817663192745, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.2587890625, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.00469970703125},
# {'layer': 17, 'dense_time': 0.007890186243057254, 'bigbird_time': 0.004673843193054199, 'longformer_time': 0.004953210878372191, 'shadowy_time': 0.0058228531169891365, 'exposer_time': 0.004135321593284605, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.26171875, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.01123046875},
# {'layer': 18, 'dense_time': 0.007910645751953126, 'bigbird_time': 0.004633088035583498, 'longformer_time': 0.004952903680801392, 'shadowy_time': 0.005838888950347899, 'exposer_time': 0.004002242527008057, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.255859375, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.005279541015625},
# {'layer': 19, 'dense_time': 0.007854551010131836, 'bigbird_time': 0.004696289291381836, 'longformer_time': 0.00493408254623413, 'shadowy_time': 0.005874667530059815, 'exposer_time': 0.004162088952064515, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.2548828125, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.0057373046875},
# {'layer': 20, 'dense_time': 0.00792504319190979, 'bigbird_time': 0.004696924180984497, 'longformer_time': 0.004988313617706299, 'shadowy_time': 0.00587706362724304, 'exposer_time': 0.004188323841094972, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.259765625, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.00506591796875},
# {'layer': 21, 'dense_time': 0.0079321497631073, 'bigbird_time': 0.00467779585838318, 'longformer_time': 0.0049527398109436025, 'shadowy_time': 0.005899714546203613, 'exposer_time': 0.004043653144836426, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.2587890625, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.00506591796875},
# {'layer': 22, 'dense_time': 0.007893524475097656, 'bigbird_time': 0.0046845132827758795, 'longformer_time': 0.004924682216644287, 'shadowy_time': 0.005894901762008666, 'exposer_time': 0.003983626232147215, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.259765625, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.004730224609375},
# {'layer': 23, 'dense_time': 0.007890759668350218, 'bigbird_time': 0.004673761262893677, 'longformer_time': 0.004930846719741821, 'shadowy_time': 0.005880811510086059, 'exposer_time': 0.003959787516593933, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.2587890625, 'longformer_sparsity': 0.330078125, 'shadowy_sparsity': 0.515625, 'exposer_sparsity': 0.0059814453125},
# ]

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
