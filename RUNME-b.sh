#!/bin/bash

set -x

sh ./scripts/overall-end2end/run.sh
echo "Script overall-end2end finish!"

sh ./scripts/overall-memory/run.sh
echo "Script overall-memory finish!"

sh ./scripts/ablation-attention/run.sh
echo "Script ablation-attention finish!"

sh ./scripts/ablation-breakdown/run.sh
echo "Script ablation-breakdown finish!"

sh ./scripts/ablation-mlp/run.sh
echo "Script ablation-mlp finish!"

sh ./scripts/ablation-operator/run.sh
echo "Script ablation-operator finish!"

sh ./scripts/scale-model/run.sh
echo "Script scale-model finish!"
