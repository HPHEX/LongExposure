import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from exposer.models.opt_peft_prefix import OPTForCausalLM
from exposer.utils.config_utils import get_opt_prefix_config
from exposer.utils.data_utils import FT_Dataset
from exposer.utils.peft_utils import mark_only_prefix_as_trainable


parser = argparse.ArgumentParser(description='Attn block sparse')
parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='model name')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--init_checkpoint', type=str, help='initial checkpoint')
parser.add_argument('--data', type=str, help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict')
parser.add_argument('--prefix_len', type=int, default=128, help='prefix length')


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    config = get_opt_prefix_config(args.model_name, args.prefix_len)
    n_layers, n_heads = config.num_hidden_layers, config.num_attention_heads

    model = OPTForCausalLM(config).to(args.device)

    from torch.optim import AdamW, lr_scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 4000, 1.0))

    if args.prefix_len > 0:
        mark_only_prefix_as_trainable(model)

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
    fwd_time = 0
    bwd_time = 0
    opt_time = 0
    all_start = torch.cuda.Event(enable_timing=True)
    all_end = torch.cuda.Event(enable_timing=True)
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)
    opt_start = torch.cuda.Event(enable_timing=True)
    opt_end = torch.cuda.Event(enable_timing=True)
    for batch_idx, data in enumerate(valid_loader):
        all_start.record()

        with torch.autocast(device_type=args.device, dtype=torch.float16):
            data = {key: value for key, value in data.items()}
            _input = data['input'].to(args.device)
            _target = data['target'].to(args.device)
            _msk = data['mask'].to(args.device)

            fwd_start.record()
            _batch, _len = _input.shape
            _lm_logits = model(_input).logits

            loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='none', label_smoothing=0.1)
            loss = loss_fct(_lm_logits.view(-1, _lm_logits.size(-1)), _target.view(-1)).view(_batch, _len)
            loss = loss * _msk
            loss = loss.sum() / (_msk.sum() + 0.0001)
            # print('loss:', loss)
            fwd_end.record()

        bwd_start.record()
        loss.backward()
        bwd_end.record()

        opt_start.record()
        optimizer.step()
        opt_end.record()

        scheduler.step()
        model.zero_grad()

        all_end.record()
        torch.cuda.synchronize()

        if batch_idx >= warmup + repitition:
            break
        if batch_idx >= warmup:
            total_time += all_start.elapsed_time(all_end)
            fwd_time += fwd_start.elapsed_time(fwd_end)
            bwd_time += bwd_start.elapsed_time(bwd_end)
            opt_time += opt_start.elapsed_time(opt_end)

    total_time /= repitition
    fwd_time /= repitition
    bwd_time /= repitition
    opt_time /= repitition
    print('{{"test_case": "Prefix", "total_time": {:.3f}, "fwd_time": {:.3f}, "bwd_time": {:.3f}, "opt_time": {:.3f}, "predict_time": 0.000}}'.format(total_time, fwd_time, bwd_time, opt_time))
