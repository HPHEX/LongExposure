import time
import argparse

import torch
import datasets
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTConfig

from exposer.models.opt_profile_attn import OPTForCausalLM
from exposer.predictors.attention_predictor_single import AttentionPredictor
from exposer.utils.config_utils import get_custom_opt_profile_config
from exposer.utils.profile_utils import traced_attn_inputs, traced_attn_scores, clear_metrics

matplotlib.use('agg')

parser = argparse.ArgumentParser(description='train predictor')
parser.add_argument('--model_name', type=str, default='facebook/opt-1.3b', choices=['facebook/opt-125m', 'facebook/opt-350m', 'facebook/opt-1.3b'], help='model name')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device for training attention predictor')
parser.add_argument('--infer_device', type=str, default='cuda:0', help='device for running model inference')
parser.add_argument('--seed', type=int, default=42, help='random seed')
# parser.add_argument('--data', type=str, help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict')

# predictor configs
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--seq_blk_size', type=int, default=0, help='sequence block size (for downsample), 0 means automatic')
parser.add_argument('--predictor_rank', type=int, default=16)
parser.add_argument('--predictor_lr', type=float, default=0.01)

# evaluation
parser.add_argument('--eval_mode', action='store_true')
parser.add_argument('--from_ckpt', type=str, help='checkpoint path')

# reference: model config --> predictor config
core_config_ref = {
    'facebook/opt-1.3b': {
        'num_hidden_layers': 24, 'hidden_size': 2048, 'ffn_dim': 8192, 'num_attention_heads': 32,
    },
    'facebook/opt-350m': {
        'num_hidden_layers': 24, 'hidden_size': 1024, 'ffn_dim': 4096, 'num_attention_heads': 16,
    },
}


def get_attention_predictor_config(model_config):
    if isinstance(model_config, OPTConfig):
        n_layers = model_config.num_hidden_layers
        n_heads = model_config.num_attention_heads
        hidden_dim = model_config.hidden_size
        return n_layers, n_heads, hidden_dim
    raise NotImplementedError


def downsample_hidden_states(hidden_states, seq_blk_size: int):
    _bsz, seq_len, _dim = hidden_states.shape
    seq_last_val = hidden_states[:, -1:, :]

    n_seq_blks = (seq_len + seq_blk_size - 1) // seq_blk_size
    padded_seq_len = n_seq_blks * seq_blk_size
    pad_seq_len = padded_seq_len - seq_len
    if pad_seq_len > 0:
        hidden_states = torch.cat(
            (hidden_states, seq_last_val.tile((1, pad_seq_len, 1))), dim=1)

    sliced = [hidden_states[:, i::seq_blk_size, :]
              for i in range(seq_blk_size)]
    return sum(sliced) / seq_blk_size


def downsample_attn_scores(attn_scores: torch.Tensor, pred_seq_len: int):
    return torch.nn.functional.adaptive_max_pool2d(attn_scores, pred_seq_len)


class MyDataset:
    def __init__(self, model_name, seq_len, batch_size, split='train', n_rows=10000):
        self.seq_len = seq_len
        self.batch_size = batch_size

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')
        all_text = ''.join(dataset[split]['text'][:n_rows])
        input_ids = tokenizer(all_text).input_ids

        blk_size = seq_len * batch_size
        n_blocks = len(input_ids) // blk_size
        input_ids = input_ids[:n_blocks * blk_size]

        self.n_blocks = n_blocks
        self.input_ids = input_ids
        self.index = 0

    def __len__(self):
        return self.n_blocks

    def __iter__(self):
        return self

    def __next__(self):
        blk_size = self.seq_len * self.batch_size
        i = self.index % self.n_blocks
        begin = i * blk_size
        input_ids = torch.LongTensor(self.input_ids[begin: begin + blk_size])
        self.index += 1
        return input_ids.view(self.batch_size, self.seq_len)


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.seq_blk_size <= 0:
        args.seq_blk_size = int(args.seq_len ** 0.5)
        assert args.seq_blk_size > 0
    pred_seq_len = (args.seq_len + args.seq_blk_size - 1) // args.seq_blk_size

    data_split = 'test' if args.eval_mode else 'train'
    dataset = MyDataset(args.model_name, args.seq_len,
                        args.batch_size, split=data_split)

    # main model
    config = get_custom_opt_profile_config(model_name=args.model_name,
                                           trace_attn_scores=True,
                                           trace_mlp_activations=False,
                                           trace_attn_inputs=True)
    # from opt_profile, not the original one from huggingface
    model = OPTForCausalLM(config)

    # copy model weights manually
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    ref_state_dict = ref_model.state_dict()
    state_dict = model.state_dict()

    for k, v in tqdm(ref_state_dict.items(), desc='copying weights'):
        if k in state_dict:
            state_dict[k].copy_(v)
        else:
            print('skip', k)
    model.to(args.infer_device)

    # prepare predictor grid
    n_layers, n_heads, hidden_dim = get_attention_predictor_config(config)
    # attn_predictors = [
    #     torch.nn.ModuleList([
    #         AttentionPredictor(hidden_dim, args.predictor_rank).to(
    #             args.device).train()
    #         for j in range(n_heads)])
    #     for i in range(n_layers)
    # ]
    # optimizers = [
    #     torch.optim.SGD(attn_predictors[i].parameters(), lr=args.predictor_lr)
    #     for i in range(n_layers)
    # ]
    if args.from_ckpt:
        attn_predictors, optimizers = torch.load(args.from_ckpt)
        for i in range(n_layers):
            for j in range(n_heads):
                optimizer = optimizers[i][j]
                for group in optimizer.param_groups:
                    group['lr'] = args.predictor_lr
    else:
        attn_predictors, optimizers = [], []
        for i in range(n_layers):
            layer_predictors, layer_optimizers = [], []
            for j in range(n_heads):
                predictor = AttentionPredictor(hidden_dim, args.predictor_rank)
                predictor = predictor.to(args.device).train()

                optimizer = torch.optim.SGD(
                    predictor.parameters(), lr=args.predictor_lr)

                layer_predictors.append(predictor)
                layer_optimizers.append(optimizer)
            attn_predictors.append(layer_predictors)
            optimizers.append(layer_optimizers)

    model.eval()
    loss_fn = torch.nn.MSELoss()
    losses = []
    for i in range(n_layers):
        layer_losses = []
        for j in range(n_heads):
            layer_losses.append([])
        losses.append(layer_losses)

    t1 = time.time()
    for step in range(args.steps):
        inputs = next(dataset).to(args.infer_device)

        clear_metrics()
        model(inputs)

        for layer, (hidden_states, attn_scores) in enumerate(zip(traced_attn_inputs, traced_attn_scores)):
            if len(attn_scores.shape) == 5:
                attn_scores = attn_scores.squeeze(dim=0)

            hidden_states = hidden_states.to(args.device)
            attn_scores = attn_scores.to(args.device)

            pred_inputs = downsample_hidden_states(
                hidden_states, args.seq_blk_size)
            pred_target = downsample_attn_scores(attn_scores, pred_seq_len)

            for head in range(n_heads):
                pred_scores = attn_predictors[layer][head](pred_inputs)
                true_scores = pred_target[:, head, :, :]
                loss = loss_fn(torch.tril(pred_scores), true_scores)

                losses[layer][head].append(loss.item())

                optimizer = optimizers[layer][head]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if layer == 0 and head == 0:
                    print(f'step {step} loss {loss}')

                if args.eval_mode:
                    fig, axs = plt.subplots(1, 2)
                    map0 = torch.tril(pred_scores[0]).detach().cpu().numpy()
                    map1 = true_scores[0].detach().cpu().numpy()
                    axs[0].imshow(map0)
                    axs[0].set_title('pred_scores')
                    axs[1].imshow(map1)
                    axs[1].set_title('true_scores')
                    plt.savefig(f'./experiments/ablation-predictor/figures/attn-layer{layer}-head{head}.pdf')
                    plt.close()

            # losses = []
            # for head in range(n_heads):
            #     pred_scores = attn_predictors[layer][head](pred_inputs)
            #     true_scores = pred_target[:, head, :, :]
            #     loss = loss_fn(torch.tril(pred_scores), true_scores)
            #     losses.append(loss)

            # optimizer = optimizers[layer]
            # loss = sum(losses) / len(losses)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # if layer == 0:
            #     print(f'layer {layer}, loss {loss}')

        if args.eval_mode:
            exit()
    t2 = time.time()
    print(f'elapsed: {t2 - t1}')

    name = args.model_name.split('/')[-1]
    save_path = f'./experiments/ablation-predictor/checkpoints/{name}_attn_predictor_r{args.predictor_rank}-0.pt'
    torch.save((attn_predictors, optimizers), save_path)

    losses = torch.tensor(losses)
    print(losses.shape)
    losses = losses.sum(dim=1)
    print(losses.shape)
    losses = losses.tolist()

    # plot losses_sum
    fig_path = f'./experiments/ablation-predictor/{name}_attn_predictor_loss.pdf'
    colors = ['#0c408c', '#204c99', '#3458a6', '#4964b3', '#5d70c0', '#717cce', 
              '#8386d7', '#8e86d2', '#9a85cc', '#a485c7', '#af85c2', '#ba84bd', 
              '#c58cbc', '#d09cc0', '#dbabc5', '#e6bbc9', '#f2ccce', '#fddcd2', 
              '#dfc4be', '#b7a2a3', '#8f8089', '#675e6e', '#3f3c54', '#171a39']
    for layer_idx, layer_losses in enumerate(losses):
        if layer_idx == 0:
            continue
        plt.plot(layer_losses, label=f'layer: {layer_idx}', color=colors[layer_idx])
    plt.savefig(fig_path)
