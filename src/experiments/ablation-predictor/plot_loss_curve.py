import matplotlib.pyplot as plt  
import re  
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cur_path', type=str, default='None', help='current path')
parser.add_argument('--output', type=str, default='None', help='output file')
args = parser.parse_args()
  
# Initialize an empty list to hold the loss values  
loss_long_exposure = []
loss_random_attn = []
loss_random_mlp = [] 
  
# Open the file and read line by line
cur_path = args.cur_path

with open(cur_path+ 'loss_opt-1.3b_exposer.log', 'r') as file:  # Replace 'loss_data.txt' with your filename  
    for line in file:  
        # Use regex to find patterns that match the loss values  
        match = re.search(r"'loss': (\d+\.\d+)", line)  
        if match:  
            # Convert the extracted loss value to a float and append to the list  
            loss_long_exposure.append(float(match.group(1)))

# Open the file and read line by line  
with open(cur_path+ 'loss_opt-1.3b_random.log', 'r') as file:  # Replace 'loss_data.txt' with your filename  
    for line in file:  
        # Use regex to find patterns that match the loss values  
        match = re.search(r"'loss': (\d+\.\d+)", line)  
        if match:  
            # Convert the extracted loss value to a float and append to the list  
            loss_random_attn.append(float(match.group(1)))

# Open the file and read line by line  
with open(cur_path+ 'loss_opt-1.3b_mlp.log', 'r') as file:  # Replace 'loss_data.txt' with your filename  
    for line in file:  
        # Use regex to find patterns that match the loss values  
        match = re.search(r"'loss': (\d+\.\d+)", line)  
        if match:  
            # Convert the extracted loss value to a float and append to the list  
            loss_random_mlp.append(float(match.group(1)))
  
print('len of loss_long_exposure:', len(loss_long_exposure))
print('len of loss_random_attn:', len(loss_random_attn))
print('len of loss_random_mlp:', len(loss_random_mlp))

len_limit = 200
loss_long_exposure = loss_long_exposure[:len_limit]
loss_random_attn = loss_random_attn[:len(loss_long_exposure)]
loss_random_mlp = loss_random_mlp[:len(loss_long_exposure)]

# Plotting

plt.figure(figsize=(3, 3))

line_colors = ['#BF84BA', '#171A39', '#FFA07A']

plt.plot(loss_random_mlp, color=line_colors[2], label='Random MLP', linewidth=2)
plt.plot(loss_random_attn, color=line_colors[1], label='Random Attention', linewidth=2)
plt.plot(loss_long_exposure, color=line_colors[0], label='Long Exposure', linewidth=2)

plt.legend(loc='upper right', fontsize=12, ncol=1, frameon=False)
plt.grid('y', linestyle='--', alpha=0.6)
 
plt.tight_layout()
plt.savefig(args.output)
