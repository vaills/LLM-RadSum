import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import csv
from nltk.translate.bleu_score import sentence_bleu
import os
from tqdm import tqdm

file_num=2

finetune_model_path = '/code/Llama-main/finetuned-model'
base_model_name_or_path = '/code/Llama-main/meta-llama/Llama-2-7b-chat-hf'


tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    base_model_name_or_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    quantization_config = BitsAndBytesConfig(
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ),
    device_map='auto'
)

model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})

model.eval()



def generate_text(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).input_ids.to('cuda')
    generate_input = {
        "input_ids": input_ids,
        "max_new_tokens": 512,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "num_beams": 3,
        "temperature": 0.1,
        "repetition_penalty": 1.3,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id
    }
    generated_ids = model.generate(**generate_input)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def generate_text2(input_text):

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        device_map='auto',
        torch_dtype=torch.float16,
        load_in_8bit=True
    )
    model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
    model.eval()


    input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).input_ids.to('cuda')
    generate_input = {
        "input_ids": input_ids,
        "max_new_tokens": 512,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "num_beams": 3,
        "temperature": 0.3,
        "repetition_penalty": 1.3,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id
    }
    generated_ids = model.generate(**generate_input)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def process_csv_and_calculate_bleu(input_csv_path, output_csv_path):
    print(input_csv_path)

    saved_line_count=0
    if os.path.exists(output_csv_path):
        with open(output_csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            saved_line_count=0
            for row in reader:
                saved_line_count+=1
            print(f'we have processed {saved_line_count} lines')



    with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        line_count=0
        for row in reader:
            line_count+=1

    with open(input_csv_path, 'r', encoding='utf-8') as csvfile, \
         open(output_csv_path, 'a', newline='', encoding='utf-8') as output_file:
        reader = csv.reader(csvfile)
        writer = csv.writer(output_file)
        
        count=0


        for row in tqdm(reader, total=line_count, desc="Processing Rows"):
 
            if saved_line_count==0 and count==0:
                count+=1

                continue
            if count<=saved_line_count: 
                count+=1
                continue                


            human, real_impression = row[0].strip().split('</s><s>Assistant: ')

            human = human.replace("<s>Human: ", "").replace("（","[").replace("）","]")
            try:
                generated_text = generate_text(f"<s>Human: {human}\n</s><s>Assistant: ")
            except Exception as e:
                generated_text = generate_text2(f"<s>Human: {human}\n</s><s>Assistant: ")
                torch.cuda.empty_cache()
                tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=False)
                tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name_or_path,
                    local_files_only=True,
                    torch_dtype=torch.float16,
                    quantization_config = BitsAndBytesConfig(
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    ),
                    device_map='auto'
                )
                model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
                model.eval()
            
            real_tokens, real_diagnosis = real_impression.split('\n诊断: ')
            real_diagnosis=real_diagnosis.replace('\n</s>','')

            generated_tokens = generated_text
            gt_impression=''
            gt_diagnosis=''
            predicted_impression=''
            predicted_diagnosis=''


            lines = real_tokens.split("\n")
            for line in lines:
                gt_impression=gt_impression+line+'\n'


            gt_diagnosis = real_diagnosis


            lines = generated_tokens.split("\n")
            save_flag=False
            for line in lines:
                if save_flag==True and "诊断" not in line:
                    predicted_impression=predicted_impression+line+'\n'
                if save_flag==True and "诊断"  in line:
                    predicted_diagnosis=line.replace('诊断: ','').replace('\n</s>','')

                if 'Assistant: ' in line:
                    predicted_impression=predicted_impression+line.replace('Assistant: ','')+'\n'
                    save_flag=True

            predicted_impression=predicted_impression.replace("[","（").replace("]","）")
            try:
                bleu_score = sentence_bleu([gt_impression], predicted_impression)
            except ValueError:  
                bleu_score = 0.0

            writer.writerow([gt_impression, predicted_impression, bleu_score,gt_diagnosis, predicted_diagnosis])


input_csv_path = f'/code/Llama-main/data/dev_only_qa_{str(file_num)}.csv'  # 或'pro_only_qa.csv'
output_csv_path = f"/code/Llama-main/results/{os.path.basename(input_csv_path)[:-4]}_predictions_{str(file_num)}.csv"

process_csv_and_calculate_bleu(input_csv_path, output_csv_path)
print(f"complete, saved file：{output_csv_path}")




