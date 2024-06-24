### DAILOUGE SUMMARIZATION ### 

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM,TrainingArguments,Trainer, GenerationConfig

DEVICE = 'cpu'

torch_device = torch.device(DEVICE)
hg_dataset = 'knkarthick/dialogsum'
dataset = load_dataset(hg_dataset)

sample_indices = [40,200]
dash_line = '-'.join('' for x in range(100))
for i, index in enumerate(sample_indices):
    print(dash_line)
    print('sample',i+1)
    print(dash_line)
    print('Input Dialog is : ')
    print(dataset['test'][index]['dialogue'])
    print(dash_line)
    print('Baseline Summary is :')
    print(dataset['test'][index]['summary'])
    print(dash_line)
    print()
    
### FLAN -T5 (Fine tuned language model) -- Text to Text Transfer Transformer ###
### also called many to many architecture--summarization, transalation, question answering. 

model_name = 'google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(torch_device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

## Inference ##
sentence = 'What time is it, Tom?'
sentence_encoded = tokenizer(sentence, return_tensors = 'pt')
sentence_decoded = tokenizer.decode(sentence_encoded['input_ids'][0])
print(f'Encoded sentence_is :{sentence_encoded['input_ids'][0]}')
print(f'Decoded sentence_is :{sentence_decoded}')

def summarize_dialogs(sample_indices,dataset,prompt = '%s'):
    for i, index in enumerate(sample_indices):
        dialog = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']
        input = prompt%(dialog)
        inputs = tokenizer(input, return_tensors='pt')
        predictions = model.generate(inputs['input_ids'],max_new_tokens=50)[0]
        output = tokenizer.decode(predictions)
        print(dash_line)
        print(f'example {i+1}')
        print(dash_line)
        print(f'Input Prompt is : {dialog}')
        print(dash_line)
        print(f'Baseline Summary is : {summary}')
        print(dash_line)
        print(f'Model Generated Output is : {output}')
        print(dash_line)
        print()
        
summarize_dialogs(sample_indices,dataset)
        
## One Shot Inference ##
def make_prompt(sample_indices, sample_index_to_summarize):
    prompt = ''
    for index in sample_indices:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']
        prompt += f""" Dialogue:\n{dialogue}\n\n what is going on ?\n{summary}\n\n\n"""  
        dialogue = dataset['test'][sample_index_to_summarize]['dialogue']
    prompt += f'Dialogue:\n{dialogue}\n\n what is going on ?\n'
    return prompt

sample_indices = [40] 
sample_index_to_summarize = 200
one_shot_prompt = make_prompt(sample_indices,sample_index_to_summarize)
print(one_shot_prompt)

summary = dataset['test'][sample_index_to_summarize]['summary']
inputs = tokenizer(one_shot_prompt,return_tensors='pt')
output = tokenizer.decode(model.generate(inputs['input_ids'],max_new_tokens = 50)[0],skip_special_tokens=True)
print(dash_line)
print(f'Baseline Summary is :\n{summary}')
print(dash_line)
print(f'Model Generation via one shot is:\n{output}')

## Few Shot Inference ##
sample_indices = [40,80,120]
sample_index_to_summarize = 200
few_shot_prompt = make_prompt(sample_indices, sample_index_to_summarize)
print(few_shot_prompt)

print(dash_line)
print(f'Baseline Summary is :\n{summary}')
print(dash_line)
print(f'Model Generation via few shot is:\n{output}')

