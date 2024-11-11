import argparse

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from exposer.models.opt_bare import OPTForCausalLM
from exposer.utils.config_utils import get_opt_config
from exposer.utils.data_utils import FT_Dataset


parser = argparse.ArgumentParser(description='Attn block sparse')
parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='model name')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--init_checkpoint', type=str, help='initial checkpoint')
parser.add_argument('--data', type=str, help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict')
parser.add_argument('--output_json', type=str, default=None, help='save')



if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    config = get_opt_config(args.model_name)
    n_layers, n_heads = config.num_hidden_layers, config.num_attention_heads

    model = OPTForCausalLM(config).to(args.device)

    from torch.optim import AdamW, lr_scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 4000, 1.0))

    with open(args.output_json, 'r') as f:
        data_re = json.load(f)
    for seq_len in [128, 256, 512, 1024, 2048, 4096]:
        try:
            valid_data = FT_Dataset(args.data, args.batch_size, seq_len)
            valid_loader = DataLoader(valid_data, 
                                    batch_size=args.batch_size, 
                                    num_workers=0, 
                                    shuffle=False, 
                                    pin_memory=True, 
                                    drop_last=False)
            
            model.train()
            with torch.autocast(device_type=args.device, dtype=torch.float16):
                for batch_idx, data in enumerate(valid_loader):
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

                    break

            max_memory = torch.cuda.max_memory_allocated()
            record = {
                "Seq_len": seq_len,
                "Full Parameter": max_memory / 1024 ** 3,
                "LoRA" : 0.0,
                "Long Exposure+LoRA": 0.0,
                "Long Exposure+LoRA (optimal)": 0.0    
            }
            data_re.append(record)
            print('{{"test_case": "Full Parameter", "max_memory_{}": {}}}'.format(seq_len, max_memory / 1024 ** 3))
        except Exception as e:
            print(e)
            continue
    with open(args.output_json, 'w') as f:
        json.dump(data_re, f)
