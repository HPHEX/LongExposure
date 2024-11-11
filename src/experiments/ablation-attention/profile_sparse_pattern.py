import argparse

import transformers
import torch
import os
from torch.utils.data import DataLoader
import numpy as np
from safetensors.torch import load_file

from exposer.models.opt_profile_attn import OPTForCausalLM
from exposer.utils.config_utils import get_opt_profile_attn_config
from exposer.utils.data_utils import FT_Dataset
from exposer.utils.profile_utils import traced_attn_scores


parser = argparse.ArgumentParser(description='Exp: profile attention scores')
parser.add_argument('--model_name', type=str, default='facebook/opt-1.3b', help='model name')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--data', type=str, help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict')


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    model_name = args.model_name.split('/')[-1]
    attn_scores_path = './src/experiments/ablation-attention/data/' + model_name + '/attn_scores.npy'
    # init_checkpoint = './checkpoints/' + model_name + '_pytorch_model.bin'
    # init_checkpoint = './reference/' + model_name + '-e2e/checkpoint/model.safetensors'
    # init_checkpoint_1 = './reference/' + model_name + '-e2e/checkpoint/model-00001-of-00002.safetensors'
    # init_checkpoint_2 = './reference/' + model_name + '-e2e/checkpoint/model-00002-of-00002.safetensors'
    
    # init_checkpoint = './reference/' + model_name + '/pytorch_model.bin'

    config = get_opt_profile_attn_config(model_name=args.model_name)
    model = OPTForCausalLM(config)
    
    hf_model = transformers.OPTForCausalLM.from_pretrained("facebook/opt-1.3b")
    state_dict = hf_model.state_dict()

    # state_dict = torch.load(init_checkpoint, map_location='cpu')
    # state_dict = load_file(init_checkpoint)
    # state_dict_1 = load_file(init_checkpoint_1)
    # state_dict_2 = load_file(init_checkpoint_2)
    # state_dict = {k: v for k, v in state_dict_1.items()}
    # state_dict.update(state_dict_2)
    # print(model.state_dict())
    for k, v in state_dict.items():
        # k = "model." + k
        if k in model.state_dict():
            # print('load', k)
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
            attn_scores = torch.cat(traced_attn_scores, dim=0).cpu().numpy()
            dir_attn_path = os.path.dirname(attn_scores_path)
            if not os.path.exists(dir_attn_path):
                os.makedirs(dir_attn_path)
            np.save(attn_scores_path, attn_scores)
            traced_attn_scores.clear()
            print('attn scores saved to:', attn_scores_path)
            exit()
