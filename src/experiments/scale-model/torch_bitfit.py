import time
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from exposer.models.gpt_bare import GPT2LMHeadModel
from exposer.utils.config_utils import get_gpt2_config
from exposer.utils.data_utils import FT_Dataset
from exposer.utils.peft_utils import mark_only_bias_as_trainable


parser = argparse.ArgumentParser(description='Attn block sparse')
parser.add_argument('--model_name', type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], help='model name')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--init_checkpoint', type=str, help='initial checkpoint')
parser.add_argument('--data', type=str, help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict')
parser.add_argument('--save_json', type=str, default=None, help='save')

if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    config = get_gpt2_config(args.model_name)
    n_layers, n_heads = config.num_hidden_layers, config.num_attention_heads

    model = GPT2LMHeadModel(config)
    model.to(args.device)

    mark_only_bias_as_trainable(model)

    from torch.optim import AdamW, lr_scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 4000, 1.0))

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
        "task" : f"a100-{model_size}-{args.seq_len}-{args.batch_size}-Bitfit",
        "time" : total_time / repitition
    }
    with open(args.save_json, 'r') as f:
        data_rc = json.load(f)
    data_rc.append(record)
    with open(args.save_json, 'w') as f:
        json.dump(data_rc, f)  
        
    print('average time:', total_time / repitition)
