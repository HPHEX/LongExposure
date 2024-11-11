#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random
import torch
import json

import torch
from torch.utils.data import Dataset


def padding_tokens(tokens, max_seq_length, pad_token, direct, max_context_length=0):
    if max_context_length == 0:
        max_context_length = max_seq_length

    if len(tokens) > max_context_length:
        if direct > 0:
            pad_tokens = tokens[:max_context_length]
        else:
            pad_tokens = tokens[-max_context_length:]
    else:
        pad_tokens = tokens
    token_len = len(pad_tokens)
    pad_tokens = pad_tokens + [pad_token for _ in range(max_seq_length - token_len)]
    return pad_tokens, token_len


class FT_Dataset(Dataset):
    def __init__(self, ft_file, batch_size, max_seq_length, 
                 max_eval_length=0, joint_lm=False, prefix_len=0, infix_len=0, 
                 prefix_cursor=1000000, infix_cursor=2000000):
        self.ft_file = ft_file
        self.ft_samples = self.read_ft_file(ft_file)[:1024]
        self.batch_size = batch_size
        self.num_examples = len(self.ft_samples)
        self.max_seq_length = max_seq_length
        self.max_eval_length = max_eval_length
        self.rng = random.Random(911)
        self.joint_lm = joint_lm
        self.num_batches = int((self.num_examples + self.batch_size - 1) / self.batch_size) 
        self.prefix_len = prefix_len
        self.infix_len = infix_len
        self.prefix_cursor = prefix_cursor
        self.infix_cursor = infix_cursor

    def __len__(self):
        return self.num_batches * self.batch_size
        
    def __getitem__(self, item):
        if(item >= self.num_examples):
            item = self.rng.randint(0, self.num_examples - 1)

        example = self.ft_samples[item]
        context = example[0]
        completion = example[1]

        pretokens = [i + self.prefix_cursor for i in range(0, self.prefix_len)] 
        intokens = [i + self.infix_cursor for i in range(0, self.infix_len)] 

        conditions = pretokens + context + intokens 
        _input, _input_len = padding_tokens(conditions + completion, self.max_seq_length, 0, 1)

        pad_targets = [0 for i in range(0, self.prefix_len)] + context + [0 for i in range(0, self.infix_len)] + completion
        _target, _ = padding_tokens(pad_targets[1:], self.max_seq_length, 0, 1)

        if not self.joint_lm:
            _msk = [0.0] * (len(conditions) - 1) + [1.0] * (_input_len - len(conditions))
        else:
            _msk = [1.0] * (_input_len - 1)

        _msk, _ = padding_tokens(_msk, self.max_seq_length, 0.0, 1)
        
        output = {}
        output["id"] = torch.tensor(item, dtype=torch.long)
        
        _query, _query_len = padding_tokens(
            conditions, self.max_seq_length, 0, -1, 
            max_context_length = self.max_seq_length - self.max_eval_length
        )
        output["query"] = torch.tensor(_query, dtype=torch.long)
        output["query_len"] = torch.tensor(_query_len, dtype=torch.long)
        output["input"] = torch.tensor(_input, dtype=torch.long) 
        output["target"] = torch.tensor(_target, dtype=torch.long) 
        output["mask"] = torch.tensor(_msk, dtype=torch.float)
        return output

    def read_ft_file(self, ft_file):
        ft_samples = []
        with open(ft_file, 'r') as reader:
            for line in reader:
                items = json.loads(line.strip())
                context = items['context']
                completion = items['completion']
                ft_samples.append([context, completion])
        return ft_samples
