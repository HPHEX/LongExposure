import argparse

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from exposer.models.opt_peft_lora import OPTForCausalLM
from exposer.utils.config_utils import get_opt_lora_config
from exposer.utils.data_utils import FT_Dataset
from exposer.utils.peft_utils import mark_only_lora_as_trainable


parser = argparse.ArgumentParser(description='Attn block sparse')
parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='model name')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--init_checkpoint', type=str, help='initial checkpoint')
parser.add_argument('--data', type=str, help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict')
parser.add_argument('--lora_dim', type=int, default=4, help='lora attn dimension')
parser.add_argument('--lora_alpha', type=int, default=32, help='lora attn alpha')
parser.add_argument('--output_json', type=str, default=None, help='save')
parser.add_argument('--card', type=str, default="a100", help='save')


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    config = get_opt_lora_config(args.model_name, args.lora_dim, args.lora_alpha)
    n_layers, n_heads = config.num_hidden_layers, config.num_attention_heads

    model = OPTForCausalLM(config)
    # if args.init_checkpoint is not None:
    #     state_dict = torch.load(args.init_checkpoint, map_location='cpu')
    #     state_dict = state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict
    #     for k, v in state_dict.items():
    #         k = 'model.' + k  if args.model_name == 'facebook/opt-350m' else k  # for opt-350m
    #         if k in model.state_dict():
    #             model.state_dict()[k].copy_(v)
    #         else:
    #             print('skip', k)
    model.to(args.device)

    from torch.optim import AdamW, lr_scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 4000, 1.0))

    if args.lora_dim > 0:
        mark_only_lora_as_trainable(model)

    valid_data = FT_Dataset(args.data, args.batch_size, args.seq_len)
    valid_loader = DataLoader(valid_data, 
                              batch_size=args.batch_size, 
                              num_workers=0, 
                              shuffle=False, 
                              pin_memory=True, 
                              drop_last=False)
    
    model.eval()
    warmup = 10
    repitition = 50
    total_time = 0
    all_start = torch.cuda.Event(enable_timing=True)
    all_end = torch.cuda.Event(enable_timing=True)
    for batch_idx, data in enumerate(valid_loader):
        all_start.record()

        with torch.autocast(device_type=args.device, dtype=torch.float16):
            data = {key: value for key, value in data.items()}
            _input = data['input'].to(args.device)
            _target = data['target'].to(args.device)
            _msk = data['mask'].to(args.device)
            _batch, _len = _input.shape
            _lm_logits = model(_input).logits

            loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='none', label_smoothing=0.1)
            loss = loss_fct(_lm_logits.view(-1, _lm_logits.size(-1)), _target.view(-1)).view(_batch, _len)
            loss = loss * _msk
            loss = loss.sum() / (_msk.sum() + 0.0001)
            # print('loss:', loss)

        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()

        all_end.record()
        torch.cuda.synchronize()

        if batch_idx >= warmup + repitition:
            break
        if batch_idx >= warmup:
            total_time += all_start.elapsed_time(all_end)

    model_size = args.model_name.split('-')[-1]
    record = {
        'task': f'{args.card}-{model_size}-{args.seq_len}-{args.batch_size}-lora',
        'time': total_time / repitition
    }
    
    with open(args.output_json, 'r') as f:
        data = json.load(f)
    data.append(record)
    with open(args.output_json, 'w') as f:
        json.dump(data, f, indent=4)
        
        
    print('average time:', total_time / repitition)
