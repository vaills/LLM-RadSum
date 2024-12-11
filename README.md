The pretrained Llama2 can be found via the following link: https://huggingface.co/meta-llama/Llama-2-7b

Model development
The script for LoRA fine-tuning based on the Llama2-7b can be found at train/sft/finetune_lora.sh, and the implementation details are located in train/sft/finetune_clm_lora.py. Fine-tuning with multiple GPUs on a single machine can be achieved by modifying the --include localhost:0 parameter in the script.

Model Testing in ./inference/
The ./inference/ directory contains scripts for model testing. These scripts enable the generation of report summaries from descriptive findings produced by different models.

Evaluation of Model Performance and Comparison in ./metrics/
The ./metrics directory includes scripts for evaluating model performance and comparing various models. These scripts provide analyses of t-test and the calculation of F1 scores.
