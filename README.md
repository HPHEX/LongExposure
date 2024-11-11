# Long Exposure

This is the artifact repository for paper #377 at SC'24, titled **Long Exposure: Accelerating Parameter-Efficient Fine-Tuning for LLMs under Shadowy Sparsity**.

This repository includes:

- **Checkpoints (`checkpoints/`):** The fine-tuned model weights for accuracy validation.
- **Dataset (`dataset/`):** The cleaned E2E dataset used for performance evaluation.
- **Log Files and Plotting Scripts (`logs/` and `plotting/`):** Files and scripts used for generating the figures in the paper.
- **Reproduced Figures (`output_figures/`):** Output directory of reproduced figures. We have provided figures (prefixed with `ref_`) in this directory that were reproduced for reference.
- **Source Code (`src/`):** The core implementation of Long Exposure.
- **Experiment Scripts (`scripts/`):** Ready-to-use scripts for running experiments corresponding to each figure and table in the paper.

## Installation

We have compiled a list of all the necessary software dependencies and their specified versions in `requirements.txt`. After installing Python (we recommend version 3.11), these dependencies can be installed automatically by executing:

```
pip install -r requirements.txt
```

**Note:** To ensure the import paths are correct, set the `PYTHONPATH` environment variable to the root directory of our repository:

```
export PYTHONPATH=<root_directory>
```

## Getting Start

Please note that reproducing all the original experiments requires strict hardware requirements: at least 1 NVIDIA A100 GPU is necessary. For reproducing experiments on scalability, 4 NVIDIA A6000 GPUs are required. To accommodate hardware limitations, we have prepared two scripts. One for quick reproduction, which plots figures from the raw data of our experiments on NVIDIA A100, and another for in-depth reproduction, which plots based on data generated from an actual run.

**Note:** We provide all the necessary files (raw logs and plotting scripts) for a quick reproduction of our results in this repository. For a more in-depth reproduction, some core components of Long Exposure are temporarily removed, as we are actively developing Long Exposure V2. This updated version will offer enhanced optimization for the PEFT process in LLMs. Both parts will be released together as a comprehensive package as soon as possible.

### 1. Quick Reproduction: Plotting from Raw Data

> **Hardware requirements: No GPUs are needed.**
>
> **Estimated Time: about 2 minites.**

To plot all figures in the evaluation section, execute the following command:

```
bash RUNME-a.sh
```

Once you have successfully run this command, all the figures will be stored in the directory `output_figures/`.

The RUNME-a.sh script reads the original log files, performs some post-processing, and plots the figures. The generated figures will be identical to those in the paper.

The matching relationship between the names of the generated figures and those in the paper is:

| Generated Figure Name | Corresponding Figure in the Paper |
| ---- | ---- |
| exp-overall-end2end-a100.png | Figure 7 (Upper) |
| exp-overall-memory-opt-350m.png | Figure 8 (Left) |
| exp-overall-memory-opt-1.3b.png | Figure 8 (Right) |
| exp-ablation-attn-sparsity.png | Figure 9 (Left Upper) |
| exp-ablation-attn-time.png | Figure 9 (Right Upper) |
| exp-ablation-mlp-sparsity.png | Figure 9 (Left Lower) |
| exp-ablation-mlp-time.png | Figure 9 (Right Lower) |
| exp-ablation-breakdown.png | Figure 10 |
| exp-ablation-predictor.png | Figure 11 (Left) |
| exp-ablation-operator-attn.png | Figure 12 (Left) |
| exp-ablation-operator-mlp.png | Figure 12 (Right) |
| exp-scale-model.png | Figure 13 |

### 2. In-depth Reproduction: Plotting from Actual Run

> **Hardware requirements: 1 NVIDIA A100 GPU.**
>
> **Estimated Time: about 1 hours.**

To reproduce all the experiments in the paper, execute the following command:

```
bash RUNME-b.sh
```

Once you have successfully run this command, all the figures will be stored in the directory `output_figures/`.

Due to fluctuations in hardware performance, the generated figures may differ slightly from those in the paper.

The matching relationship between the names of the generated figures and those in the paper is the same as the table above.

**Note:** To shorten the reproduction time, we did not include Figure 11 and Table IV from the paper in the scripts, as they involve the entire fine-tuning process over the whole dataset. We also separated the reproduction of Figure 14 from the `RUNME-b.sh` because it requires a multi-GPU server with at least 4 NVIDIA A6000. However, we have prepared separate scripts for reproducing them.

1. **Figure 11 Reproduction.**

> **Hardware requirements: 1 NVIDIA A100 GPU.**
> 
> **Estimated Time: about 6 hours.**

To reproduce Figure 11(Left), execute the following command:

```
bash ./scrips/ablation-predictor/run.sh
```

Once you have successfully run this command, you will get a figure with three loss curves stored in the directory `output_figures/`.

We decided not to reproduce Figure 11 (Right) because it merely visualizes a few prediction results, while the accuracy of the predictor has already been demonstrated by Figure 11 (Left).

2. **Table IV Reproduction.**

> **Hardware requirements: 1 NVIDIA A100 GPU.**
> 
> **Estimated Time: about 6 hours.**

We provide the fine-tuned model weights in the checkpoints/ directory, allowing you to directly perform inference on downstream tasks for evaluation.

Due to the large size of the model weights, we have split them into multiple sub-units and used Git LFS to upload them to GitHub.

To retrieve the model weights, first execute the following command to pull the large files:

```
git lfs pull
```

Next, run the following scripts to concatenate the sub-units:

```
cd checkpoints
bash cat_tensor.sh
```

We use the framework lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness) from EleutherAI to simplify the evaluation, which can be installed by:

```
pip install lm-evel
```

To evaluate OPT-350M, execute the following command:

```
# Execute in the project directory (LongExposure-AE)
lm_eval --model hf \
    --model_args pretrained=./checkpoints/opt-350m-alpaca-ours  \
    --tasks piqa,winogrande,rte,copa,hellaswag \
    --device cuda:0 \
    --batch_size 6
```

The accuracy of each downstream tasks will output to the console. Adjust the `batch_size` parameter to fit the memory capacity of your device.

Similarly, to evaluate OPT-1.3B and OPT-2.7B, execute the following commands:

```
# Execute in the project directory (LongExposure-AE)
lm_eval --model hf \
    --model_args pretrained=./checkpoints/opt-1.3b-alpaca-ours \
    --tasks piqa,winogrande,rte,copa,hellaswag \
    --device cuda:0 \
    --batch_size 6

lm_eval --model hf \
    --model_args pretrained=./checkpoints/opt-2.7b-alpaca-ours \
    --tasks piqa,winogrande,rte,copa,hellaswag \
    --device cuda:0 \
    --batch_size 6
```

**Note:** In fact, we also provide fine-tuning scripts to obtain these model weights, located in the directory `./src/experiments/overall-accuracy/`. The script `finetune.py` controls the fine-tuning process, while `merge_and_save.py` handles the merging and saving of LoRA weights. However, executing these scripts can take tens of hours. Therefore, we have provided the fine-tuned model weights and included these scripts just for completeness.

3. **Figure 14 Reproduction.**

> **Hardware requirements: 4 NVIDIA A6000 GPUs.**
> 
> **Estimated Time: about 10 minutes.**

We enable distributed model training with the framework DeepSpeed (https://github.com/microsoft/DeepSpeed) from Microsoft, which can be installed by:

```
pip install deepspeed
```

To reproduce Figure 14, execute the following command:

```
bash ./scrips/scale-card/run.sh
```

Once you have successfully run this command, you will get the resulting figure stored in the directory `output_figures/`.

### 3. Detailed Reproduction: Plotting for Each Experiments.

We provide a single script for each experiment in the directory `scripts/`.

The correspondence between the names of the generated figures and those in the paper is detailed in the table above.

To reproduce a specific experiment, locate the corresponding subdirectory and execute the script within it. The resulting figure will be stored in the directory `output_figures/`.
