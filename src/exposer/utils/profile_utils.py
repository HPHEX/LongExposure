# Tensor
traced_attn_scores = []
traced_mlp_activations = []

traced_attn_inputs = []  # self-attention inputs (hidden_states)
traced_mlp_inputs = []  # mlp inputs (hidden_states)

# Time
metrics_attn_predictor_time = []
metrics_mlp_predictor_time = []

def clear_metrics():
    traced_attn_scores.clear()
    traced_mlp_activations.clear()
    traced_attn_inputs.clear()
    traced_mlp_inputs.clear()
    metrics_attn_predictor_time.clear()
    metrics_mlp_predictor_time.clear()
