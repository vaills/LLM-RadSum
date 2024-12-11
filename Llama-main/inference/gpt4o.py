import pandas as pd
from pathlib import Path
from tqdm import tqdm
from openpyxl import load_workbook
from openai import OpenAI
import sys
sys.path.append("..")
from pathlib import Path
from tqdm import tqdm
import json
import openai
import time
import pandas as pd
import os


root = 'E:/Papers/LLM-RadSum/human_eval/'
dataset_name = ''
reports_file = Path(root) / f'{data_name}_selected_reports.xlsx'
output_file = Path(root) / f'{data_name}_selected_reports_gpt4o.xlsx'

df = pd.read_excel(reports_file)
selected_columns = df.iloc[:, [0, 1, 2, 3, 4, 5, 16, 17]].copy()

client = OpenAI(api_key="") # your key

def gpt_generation(radiology_findings):
  completion = client.chat.completions.create(
    model="gpt-4o", 
    messages=[
      {"role": "system", "content": "你是一名放射医生，你要对下面自由撰写的影像学findings进行impression的撰写。impression要分点展示。impression之后要换行进行病人的诊断，只需要输出诊断关键词。"},
      {"role": "user", "content": f"{radiology_findings}"}

    ]
  )
  return completion.choices[0].message

def read_existing_results(file):
    try:
        return pd.read_excel(file)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        return pd.DataFrame()

def append_to_excel(file, df_row):
    try:
        if not Path(file).exists():
            df_row.to_excel(file, index=False)
        else:
            with pd.ExcelWriter(file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                df_row.to_excel(writer, startrow=writer.sheets['Sheet1'].max_row, header=False, index=False)
    except Exception as e:
        print(f"Error saving to {file}: {e}")

existing_results = read_existing_results(output_file)

if not existing_results.empty:
    processed_indices = existing_results.index.tolist()
else:
    processed_indices = []

for index, row in tqdm(selected_columns.iterrows(), total=len(selected_columns), desc="Processing rows"):
    if index in processed_indices:
        continue 
    
    radiology_findings = row[6]
    gpt4_result = gpt_generation(radiology_findings)
    
    new_row = pd.DataFrame([row.tolist() + [gpt4_result]], columns=selected_columns.columns.tolist() + ['GPT4_Result'])

    existing_results = pd.concat([existing_results, new_row])
    
    append_to_excel(output_file, new_row)

print("Processing complete.")
