import matplotlib.pyplot as plt


data = [
    # {"layer": 0, "recall":  0.0},
    # {"layer": 1, "recall":  0.0},
    # {"layer": 2, "recall":  0.0},
    # {"layer": 3, "recall":  0.0},
    # {"layer": 4, "recall":  0.0},
    # {"layer": 5, "recall":  0.0},
    # {"layer": 6, "recall":  0.0},
    # {"layer": 7, "recall":  0.0},
    # {"layer": 8, "recall":  0.0},
    # {"layer": 9, "recall":  0.0},
    # {"layer": 10, "recall": 0.0},
    # {"layer": 11, "recall": 0.0},
    # {"layer": 12, "recall": 0.0},
    # {"layer": 13, "recall": 0.0},
    # {"layer": 14, "recall": 0.0},
    # {"layer": 15, "recall": 0.0},
    # {"layer": 16, "recall": 0.0},
    # {"layer": 17, "recall": 0.0},
    # {"layer": 18, "recall": 0.0},
    # {"layer": 19, "recall": 0.0},
    # {"layer": 20, "recall": 0.0},
    # {"layer": 21, "recall": 0.0},
    # {"layer": 22, "recall": 0.0},
    # {"layer": 23, "recall": 0.0},
]

# print average recall
recall_sum = 0
for d in data:
    recall_sum += d['recall']
print('Average recall:', recall_sum / len(data))
exit()

x = [d['layer'] for d in data]
y = [d['recall'] for d in data]

plt.figure(figsize=(5, 4))
colors = ['#0c408c', '#204c99', '#3458a6', '#4964b3', '#5d70c0', '#717cce', 
            '#8386d7', '#8e86d2', '#9a85cc', '#a485c7', '#af85c2', '#ba84bd', 
            '#c58cbc', '#d09cc0', '#dbabc5', '#e6bbc9', '#f2ccce', '#fddcd2', 
            '#dfc4be', '#b7a2a3', '#8f8089', '#675e6e', '#3f3c54', '#171a39']
plt.bar(x, y, color=colors, edgecolor='black', label=x)
plt.xlabel('Layer')
plt.ylabel('Recall')
plt.ylim(0.0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0), frameon=False)

plt.tight_layout()
plt.savefig('./experiments/ablation-predictor/opt_1.3b_mlp_predictor_recall.pdf')
