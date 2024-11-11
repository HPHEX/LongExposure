#!/bin/bash

# Define an array of models
models=(
  "opt-1.3b-alpaca-ours"
  "opt-1.3b-alpaca-base"
  "opt-2.7b-alpaca-ours"
  "opt-2.7b-alpaca-base"
)

# Iterate through each model
for model in "${models[@]}"; do
  # Change to the model directory
  cd ${model}
  
  # Find unique base names for the current model parts
  parts_base_names=$(ls model-*_part_* 2>/dev/null | sed -E 's/_part_[a-z]+//g' | sort | uniq)
  
  # Iterate through each base name
  for base_name in $parts_base_names; do
    # Find all parts for the current base name
    parts=($(ls ${base_name}_part_* 2>/dev/null))
    
    # If there are parts found
    if [ ${#parts[@]} -gt 0 ]; then
      # Define the output file
      output_file="${base_name}.safetensors"
      
      # Concatenate all parts into the output file
      cat "${parts[@]}" > "$output_file"
      
      echo "Merged parts into ${output_file}"
    else
      echo "No parts found for ${base_name}"
    fi
  done

  # Go back to the checkpoints directory
  cd ..
done
