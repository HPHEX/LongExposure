import time
import argparse

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from exposer.models.opt_profile_mlp import OPTForCausalLM
from exposer.predictors.mlp_predictor import MLPPredictor
from exposer.utils.data_utils import FT_Dataset
from exposer.utils.config_utils import get_opt_profile_mlp_config
from exposer.utils.profile_utils import traced_mlp_activations, traced_mlp_inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train predictor')
    parser.add_argument('--model_name', type=str, default='facebook/opt-1.3b', help='model name')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict')
    # predictor configs
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--value_threshold', type=float, default=0.1)
    parser.add_argument('--ratio_threshold', type=float, default=0.003)
    parser.add_argument('--neuron_blk_size', default=128, type=int)
    parser.add_argument('--predictor_lr', type=float, default=0.001)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model_name = args.model_name
    config = get_opt_profile_mlp_config(model_name)
    model = OPTForCausalLM(config)

    ckpt_path = f'./checkpoints/{model_name.split("/")[-1]}_pytorch_model.bin'
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(args.device)

    valid_data_path = f'./dataset/e2e/valid_opt.jsonl'
    valid_data = FT_Dataset(valid_data_path, args.batch_size, args.seq_len)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)

    mlp_predictors = []
    predictor_optimizers = []
    predictor_losses = []

    n_layers, hidden_dim = config.num_hidden_layers, config.hidden_size

    if args.eval is False:
        for i in range(n_layers):
            _predictor = MLPPredictor(hidden_dim, hidden_dim * 4, args.neuron_blk_size)
            _optimizer = torch.optim.Adam(_predictor.parameters(), lr=args.predictor_lr)
            mlp_predictors.append(_predictor)
            predictor_optimizers.append(_optimizer)
            predictor_losses.append([])

        model.eval()
        loss_fn = torch.nn.BCELoss()

        start_time = time.time()

        for batch_idx, data in enumerate(valid_loader):
            if batch_idx >= args.steps:
                break

            print(f'batch: {batch_idx}')
            
            data = {key: value for key, value in data.items()}
            _input = data['input'].to(args.device)
            model(_input)

            for layer_idx, (_inputs, _activations) in enumerate(zip(traced_mlp_inputs, traced_mlp_activations)):
                _activations = (_activations > args.value_threshold).float()
                _target = _activations.view(_activations.size(0), -1, args.neuron_blk_size).mean(dim=-1)

                _predictor = mlp_predictors[layer_idx]
                _optimizer = predictor_optimizers[layer_idx]

                _optimizer.zero_grad()
                _inputs = _inputs.to(args.device)
                _target = _target.to(args.device)
                _predictor.to(args.device)
                _output = _predictor(_inputs)
                _loss = loss_fn(_output, _target)
                _loss.backward()
                _optimizer.step()

                predictor_losses[layer_idx].append(_loss.item())
                # print(f'layer: {layer_idx}, loss: {_loss.item()}')

            traced_mlp_inputs.clear()
            traced_mlp_activations.clear()

        end_time = time.time()
        print(f'time: {end_time - start_time}')

        # plot loss
        fig_path = f'./experiments/ablation-predictor/{model_name.split("/")[-1]}_mlp_predictor_loss.pdf'
        colors = ['#0c408c', '#204c99', '#3458a6', '#4964b3', '#5d70c0', '#717cce', 
                  '#8386d7', '#8e86d2', '#9a85cc', '#a485c7', '#af85c2', '#ba84bd', 
                  '#c58cbc', '#d09cc0', '#dbabc5', '#e6bbc9', '#f2ccce', '#fddcd2', 
                  '#dfc4be', '#b7a2a3', '#8f8089', '#675e6e', '#3f3c54', '#171a39']
        for layer_idx, layer_losses in enumerate(predictor_losses):
            plt.plot(layer_losses, label=f'layer: {layer_idx}', color=colors[layer_idx])
        plt.savefig(fig_path)

        # save predictor
        for layer_idx, predictor in enumerate(mlp_predictors):
            predictor_path = f'./experiments/ablation-predictor/checkpoints/{model_name.split("/")[-1]}_mlp_predictor_{layer_idx}.pt'
            torch.save(predictor.state_dict(), predictor_path)

    else:
        for i in range(n_layers):
            _predictor = MLPPredictor(hidden_dim, hidden_dim * 4, args.neuron_blk_size)
            _predictor_path = f'./experiments/ablation-predictor/checkpoints/{model_name.split("/")[-1]}_mlp_predictor_{i}.pt'
            _predictor.load_state_dict(torch.load(_predictor_path, map_location='cpu'))
            mlp_predictors.append(_predictor)

    # evaluate predictor
    test_data_path = f'./dataset/e2e/test_opt.jsonl'
    test_data = FT_Dataset(test_data_path, args.batch_size, args.seq_len)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)

    model.eval()
    for predictor in mlp_predictors:
        predictor.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = {key: value for key, value in data.items()}
            _input = data['input'].to(args.device)
            model(_input)

            for layer_idx, (_inputs, _activations) in enumerate(zip(traced_mlp_inputs, traced_mlp_activations)):
                _activations = (_activations > args.value_threshold).float()
                _target = _activations.view(_activations.size(0), -1, args.neuron_blk_size).mean(dim=-1)

                _predictor = mlp_predictors[layer_idx]
                _inputs = _inputs.to(args.device)
                _target = _target.to(args.device)
                _predictor.to(args.device)
                _output = _predictor(_inputs)

                _target = (_target > args.ratio_threshold)
                _output = (_output > args.ratio_threshold)
                _recall = (_output & _target).sum() / _target.sum()

                print(f'{{"layer": {layer_idx}, "recall": {_recall.item()}}}')

            break
