The pretrained Llama2 can be found via the following link: https://huggingface.co/meta-llama/Llama-2-7b

LoRA Fine-Tuning
The script for LoRA fine-tuning can be found at train/sft/finetune_lora.sh, and the implementation details are located in train/sft/finetune_clm_lora.py. Fine-tuning with multiple GPUs on a single machine can be achieved by modifying the --include localhost:0 parameter in the script.

