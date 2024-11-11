import numpy as np
import matplotlib.pyplot as plt


sparsity_block = 128
sparsity_threshold = 0.9

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_name', type=str, default='facebook/opt-1.3b', help='model name')
    argparser.add_argument('--output', type=str, default=None )
    argparser.add_argument('--data_path', type=str, default="./src/experiments/ablation-mlp/data/")
    args = argparser.parse_args()
    model_name = args.model_name
    model_name = model_name.split('/')[-1]

    data_path = args.data_path+ model_name + "/mlp_activations.npy"
    figures_path = "./src/experiments/ablation-mlp/figures/" + model_name

    mlp_activations = np.load(data_path)
    _layers, _hidden_size = mlp_activations.shape
    print(mlp_activations.shape)

    data_element_sparsity = []
    data_element_sparsity_percent_0 = []
    data_exposer_sparsity_percent_1 = []
    data_exposer_sparsity_percent_3 = []
    data_exposer_sparsity_percent_5 = []

    for i in range(_layers):
        print(f"now is processing layer {i} of total {_layers}")
        _activations = mlp_activations[i]
        # calculate the sparsity of the activations
        element_sparsity = np.sum(_activations == 0) / _hidden_size
        # print('Layer', i, 'sparsity:', sparsity)
        data_element_sparsity.append(element_sparsity)

        num_sparsity_blocks_percent_0 = 0
        num_sparsity_blocks_percent_1 = 0
        num_sparsity_blocks_percent_3 = 0
        num_sparsity_blocks_percent_5 = 0
        num_blocks = _hidden_size // sparsity_block
        for j in range(0, _hidden_size, sparsity_block):
            blk_activations = _activations[j:j+sparsity_block]
            max_val = np.max(blk_activations)
            # print('Layer', i, 'Block', j, 'max_val:', max_val)
            blk_sparsity_0 = np.sum(blk_activations == 0) / sparsity_block
            if blk_sparsity_0 > sparsity_threshold:
                num_sparsity_blocks_percent_0 += 1
            blk_sparsity_10 = np.sum(blk_activations < 0.01 * max_val) / sparsity_block
            if blk_sparsity_10 > sparsity_threshold:
                num_sparsity_blocks_percent_1 += 1
            blk_sparsity_30 = np.sum(blk_activations < 0.03 * max_val) / sparsity_block
            if blk_sparsity_30 > sparsity_threshold:
                num_sparsity_blocks_percent_3 += 1
            blk_sparsity_50 = np.sum(blk_activations < 0.05 * max_val) / sparsity_block
            if blk_sparsity_50 > sparsity_threshold:
                num_sparsity_blocks_percent_5 += 1
        data_element_sparsity_percent_0.append(num_sparsity_blocks_percent_0 / num_blocks)
        data_exposer_sparsity_percent_1.append(num_sparsity_blocks_percent_1 / num_blocks)
        data_exposer_sparsity_percent_3.append(num_sparsity_blocks_percent_3 / num_blocks)
        data_exposer_sparsity_percent_5.append(num_sparsity_blocks_percent_5 / num_blocks)

    # # only keep the even layers
    # data_element_sparsity = data_element_sparsity[::2]
    # data_element_sparsity_percent_0 = data_element_sparsity_percent_0[::2]
    # data_exposer_sparsity_percent_1 = data_exposer_sparsity_percent_1[::2]
    # data_exposer_sparsity_percent_3 = data_exposer_sparsity_percent_3[::2]
    # data_exposer_sparsity_percent_5 = data_exposer_sparsity_percent_5[::2]

    data_element_sparsity = data_element_sparsity
    data_element_sparsity_percent_0 = data_element_sparsity_percent_0
    data_exposer_sparsity_percent_1 = data_exposer_sparsity_percent_1
    data_exposer_sparsity_percent_3 = data_exposer_sparsity_percent_3
    data_exposer_sparsity_percent_5 = data_exposer_sparsity_percent_5

    # Setting up the bar width
    bar_width = 0.15 
    
    # Setting the position of the bars on the x-axis  
    # r1 = np.arange(_layers // 2)
    # r1 = np.arange(_layers)
    r1 = np.arange(_layers)
    r2 = [x + bar_width for x in r1]  
    r3 = [x + bar_width for x in r1]  
    r4 = [x + bar_width for x in r1]  
    r5 = [x + bar_width for x in r1]

    print(r1)
    print(len(data_element_sparsity))
    # Creating the figure
    plt.figure(figsize=(4, 3))
    
    # Creating the bar plot
    line_colors = ['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#171A39']
    plt.plot(r1, data_element_sparsity, color=line_colors[0], marker='o', label='Shadowy', markersize=3, linewidth=1)
    plt.plot(r2, data_element_sparsity_percent_0, color=line_colors[1], marker='^', label='0%', markersize=3, linewidth=1)
    plt.plot(r3, data_exposer_sparsity_percent_1, color=line_colors[2], marker='s', label='1%', markersize=3, linewidth=1)
    plt.plot(r4, data_exposer_sparsity_percent_3, color=line_colors[3], marker='d', markersize=3, linewidth=1)
    plt.plot(r5, data_exposer_sparsity_percent_5, color=line_colors[4], marker='x', markersize=3, linewidth=1)
    
    print("here!!")
    # Adding labels  
    plt.xlabel('Layer')  
    plt.ylabel('Sparsity Ratio')
    # plt.xticks([r + bar_width for r in range(_layers // 2)], range(0, _layers, 2))
    plt.xticks([r + bar_width for r in range(_layers)], range(0, _layers), fontsize=10, rotation=90)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=10)

    # Creating legend & title for the bar plot  
    # plt.legend(loc='upper right', fontsize=10, frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.5)

    # Saving the figure (optional)  
    plt.tight_layout()
    plt.savefig(args.output)
