import argparse

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import deepspeed

from exposer.models.opt_exposer_bitfit import OPTForCausalLM
from exposer.utils.config_utils import get_opt_exposer_bitfit_config
from exposer.utils.data_utils import FT_Dataset
from exposer.utils.peft_utils import mark_only_bias_as_trainable


parser = argparse.ArgumentParser(description='Attn block sparse')
parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='model name')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--init_checkpoint', type=str, help='initial checkpoint')
parser.add_argument('--data', type=str, help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict')
parser.add_argument('--sparsity_config', type=str, default='bigbird', choices=['bigbird', 'longformer', 'fixed', 'variable', 'local'], help='sparsity config')
parser.add_argument('--mlp_block_size', type=int, default=128, help='mlp block size')
parser.add_argument('--mlp_threshold', type=float, default=0.80, help='mlp threshold')
parser.add_argument('--local_rank', type=int, default=-1, help='local rank')
parser.add_argument('--save_json', type=str, help='save json file')
parser.add_argument('--num_cards', type=int, default=1, help='number of cards')


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)

    # 0. Set CUDA_VISIBLE_DEVICES
    # 1. Initialize DeepSpeed
    deepspeed.init_distributed(dist_backend='nccl')

    # 2. Get deepspeed config
    deepspeed_config = {
        # 'train_batch_size': args.batch_size,
        'train_micro_batch_size_per_gpu': args.batch_size,
        'gradient_accumulation_steps': 1,
        'gradient_clipping': 1.0,
        'fp16': {
            'enabled': True,
            'loss_scale': 0,
            'initial_scale_power': 16,
            'loss_scale_window': 1000,
            'hysteresis': 2,
            'min_loss_scale': 1
        },
        'zero_optimization': {
            'stage': 0,
            'cpu_offload': False,
        },
        'wall_clock_breakdown': True,
    }

    config = get_opt_exposer_bitfit_config(args.model_name, 
                                           sparse_config=args.sparsity_config, 
                                           mlp_block_size=args.mlp_block_size, 
                                           mlp_threshold=args.mlp_threshold)
    n_layers, n_heads = config.num_hidden_layers, config.num_attention_heads

    model = OPTForCausalLM(config)
    model.to(args.device)

    from torch.optim import AdamW, lr_scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 4000, 1.0))

    # 3. Initialize model_engine
    model_engine, optimizer, _, _ = deepspeed.initialize(args=args, 
                                                         model=model, 
                                                         model_parameters=model.parameters(),
                                                         optimizer=optimizer,
                                                         lr_scheduler=scheduler,
                                                         config=deepspeed_config)

    mark_only_bias_as_trainable(model)

    num_gpus = torch.cuda.device_count()
    train_batch_size = args.batch_size * num_gpus
    valid_data = FT_Dataset(args.data, train_batch_size, args.seq_len)
    valid_loader = DataLoader(valid_data, 
                              batch_size=train_batch_size, 
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
    all_start.record()
    for batch_idx, data in enumerate(valid_loader):

        model_engine.train()
        data = {key: value.to(args.device) for key, value in data.items()}
        _input = data['input']
        _target = data['target']
        _msk = data['mask']

        _batch, _len = _input.shape
        _lm_logits = model_engine(_input).logits

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='none', label_smoothing=0.1)
        loss = loss_fct(_lm_logits.view(-1, _lm_logits.size(-1)), _target.view(-1)).view(_batch, _len)
        loss = loss * _msk
        loss = loss.sum() / (_msk.sum() + 0.0001)
        # print('loss:', loss)

        model_engine.backward(loss)
        model_engine.step()

        all_end.record()
        torch.cuda.synchronize()

    all_end.record()
    torch.cuda.synchronize()

    with open(args.save_json, 'r') as f:
        data_re = json.load(f)
    
    record = {
        "task" : 'exposer + bitfit',
        "num_cards" : args.num_cards,
        "time" : all_start.elapsed_time(all_end)
    }
    if record['task'] not in [d['task'] for d in data_re]:
        data_re.append(record)
    with open(args.save_json, 'w') as f:
        json.dump(data_re, f, indent=4)
    print('total time:', all_start.elapsed_time(all_end))
