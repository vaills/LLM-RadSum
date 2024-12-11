import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig
finetune_model_path='/zhengsunyi/code/Llama-Chinese-main/model_Llama2-Chinese-7b-Chat-LoRA'  
config = PeftConfig.from_pretrained(finetune_model_path)
base_model_name_or_path='/zhengsunyi/code/Llama-Chinese-main/meta-llama/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path,device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
model =model.eval()
findings=''
input_ids = tokenizer([f'<s>Human: according to radiology findings {radiology findings}, help me to write the impression+\n+</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')        
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
