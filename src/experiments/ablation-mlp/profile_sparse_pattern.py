import argparse

import transformers
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

from exposer.models.opt_profile_mlp import OPTForCausalLM
from exposer.utils.config_utils import get_opt_profile_mlp_config
from exposer.utils.data_utils import FT_Dataset
from exposer.utils.profile_utils import traced_mlp_activations


parser = argparse.ArgumentParser(description='Exp: profile mlp activations')
parser.add_argument('--model_name', type=str, default='facebook/opt-1.3b', help='model name')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--data', type=str, help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict')


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    model_name = args.model_name.split('/')[-1]
    # init_checkpoint = './reference/' + model_name + '/pytorch_model.bin'
    hf_model = transformers.OPTForCausalLM.from_pretrained('facebook/opt-1.3b')
    
    mlp_activations_path = './src/experiments/ablation-mlp/data/' + model_name + '/mlp_activations.npy'

    config = get_opt_profile_mlp_config(model_name=args.model_name)
    model = OPTForCausalLM(config)
    state_dict = hf_model.state_dict()
    # state_dict = torch.load(init_checkpoint, map_location='cpu')
    for k, v in state_dict.items():
        if k in model.state_dict():
            model.state_dict()[k].copy_(v)
        else:
            print('skip', k)
    model.to(args.device)
    print('model name:', model_name)

    dataset = FT_Dataset(args.data, args.batch_size, args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print('data length:', len(dataloader))

    model.eval()
    with torch.autocast(device_type=args.device, dtype=torch.float16):
        for idx, data in enumerate(dataloader):
            data = {k: v for k, v in data.items()}
            _input = data['input'].to(args.device)
            model(_input)
            # save metrics to file
            mlp_activations = torch.cat(traced_mlp_activations, dim=0).cpu().numpy()
            dir_mlp_path = os.path.dirname(mlp_activations_path)
            if not os.path.exists(dir_mlp_path):
                os.makedirs(dir_mlp_path)
            np.save(mlp_activations_path, mlp_activations)
            traced_mlp_activations.clear()
            exit()
