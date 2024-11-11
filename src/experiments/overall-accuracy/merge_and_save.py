import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

from exposer.models import opt_peft_lora_merge
from exposer.utils.config_utils import get_custom_opt_config


def load_initial_weights(model, src_state_dict):
    state_dict = model.state_dict()
    for k, v in src_state_dict.items():
        if k in state_dict:
            # print('load', k)
            state_dict[k].copy_(v)
        else:
            print('skip', k)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    model_name = args.model_name
    ckpt_path = args.ckpt_path
    output_dir = args.output_dir
    
    config = get_custom_opt_config(model_name)
    config.merge_lora_weights = True # extra config to merge LoRA weights
    lora_model = opt_peft_lora_merge.OPTForCausalLM(config)

    if model_name == 'facebook/opt-350m':
        init_checkpoint = os.path.join(ckpt_path, 'model.safetensors')
        state_dict = load_file(init_checkpoint)
        load_initial_weights(lora_model, state_dict)
    elif model_name == 'facebook/opt-1.3b':
        init_checkpoint_1 = os.path.join(ckpt_path, 'model-00001-of-00002.safetensors')
        init_checkpoint_2 = os.path.join(ckpt_path, 'model-00002-of-00002.safetensors')
        state_dict_1 = load_file(init_checkpoint_1)
        state_dict_2 = load_file(init_checkpoint_2)
        state_dict = {k: v for k, v in state_dict_1.items()}
        state_dict.update(state_dict_2)
        load_initial_weights(lora_model, state_dict)
    elif model_name == 'facebook/opt-2.7b':
        init_checkpoint_1 = os.path.join(ckpt_path, 'model-00001-of-00003.safetensors')
        init_checkpoint_2 = os.path.join(ckpt_path, 'model-00002-of-00003.safetensors')
        init_checkpoint_3 = os.path.join(ckpt_path, 'model-00003-of-00003.safetensors')
        state_dict_1 = load_file(init_checkpoint_1)
        state_dict_2 = load_file(init_checkpoint_2)
        state_dict_3 = load_file(init_checkpoint_3)
        state_dict = {k: v for k, v in state_dict_1.items()}
        state_dict.update(state_dict_2)
        state_dict.update(state_dict_3)
        load_initial_weights(lora_model, state_dict)
    else:
        print('model_name not supported')
        raise NotImplementedError

    lora_model.eval() # trigger weights merge
    
    merged_model = AutoModelForCausalLM.from_config(config)
    load_initial_weights(merged_model, lora_model.state_dict())
    merged_model.save_pretrained(output_dir)

    # save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
