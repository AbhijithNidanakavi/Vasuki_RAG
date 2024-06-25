### FINE TUNING ### 

import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM,TrainingArguments,Trainer, GenerationConfig

DEVICE = 'cpu'

torch_device = torch.device(DEVICE)
hg_dataset = 'knkarthick/dialogsum'
dataset = load_dataset(hg_dataset)

model_name = 'google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(torch_device)
tokenizer =  AutoTokenizer.from_pretrained(model_name)

def number_of_trainable_model_parameters(model):
    all_model_params = 0
    trainable_model_params = 0
    for _,param in model.named_parameters():
        all_model_params = all_model_params + param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    result = f'All Model Parameters are : {all_model_params}\n'
    result += f'Trainable Model Parameters are : {trainable_model_params}\n'
    result += f'Percentage of Model Parameters are : {(trainable_model_params/all_model_params)*100}'
    return result
print(number_of_trainable_model_parameters(model))

## Testing the Model with Zero_Shot_Inference ##

index = 200
dialogue = dataset['test'][index]['dialogue']        
summary = dataset['test'][index]['summary']
prompt = f"""Summarize the following conversation : 
{dialogue}
summary:
"""
inputs =  tokenizer(prompt, return_tensors='pt')
output =  tokenizer.decode(
    model.generate(
        inputs['input_ids'],
        max_new_tokens = 200
    )[0],
    skip_special_tokens=True)
dash_line = '-'.join('' for x in  range (100))
print(dash_line)
print(f'Input Prompt is :\n{prompt}')
print(dash_line)
print(f'Baseline Summary is :\n{summary}')
print(dash_line)
print(f'Model Generated Zero_shot_Inference is :\n{output}')

## Performing Full FineTuning ##

def tokenize_function(sample):
    start_prompt = 'Summarize the following conversation'
    end_prompt = 'Summary'
    prompt = [start_prompt + dialogue + end_prompt for dialouge in sample['dialogue']]
    sample['input_ids'] = tokenizer(prompt,padding='max_length',truncation=True,return_tensors='pt').input_ids
    sample['labels'] = tokenizer(sample['summary'],padding='max_length',truncation=True,return_tensors='pt').input_ids
    return sample

## Calling the above function containing 3 different splits : Train, Validation & Test ##

tokenized_datasets = dataset.map(tokenize_function,batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id','topic','dialogue','summary'])

tokenized_datasets = tokenized_datasets.filter(lambda sample, index: index%100== 0,with_indices=True)
print(f'Shapes of the datasets : ')
print(f'Training data : {tokenized_datasets['train'].shape}')
print(f'Validation data : {tokenized_datasets['validation'].shape}')
print(f'Test data : {tokenized_datasets['test'].shape}')
print(tokenized_datasets)

## Fine Tuning the model with preprocessed dataset ##
## By utilizing the builtin hugginface trainer class ##

output_directory = f'./Dialogue-Summary-Training-{str(int(time.time()))}'

training_args = TrainingArguments(
                output_dir=output_directory,
                learning_rate=1e-5, 
                num_train_epochs=1,
                weight_decay=0.01,
                #max_steps=1,
                logging_steps=1)

trainer = Trainer(model=model,
                  args = training_args,
                  train_dataset=tokenized_datasets['train'],
                  eval_dataset=tokenized_datasets['validation'],
                  )
trainer.train()

instruct_model = AutoModelForSeq2SeqLM.from_pretrained('full/').to(torch_device)
model = model.to(torch_device)

# Evaluating the model qualitatively #

index = 200
dialogue = dataset['test'][index]['dialogue']
human_baseline_summary = dataset['test'][index]['summary']
prompt = f"""Summarize the following conversation 
{dialogue}
summary:"""
input_ids = tokenizer(prompt, return_tensors='pt').input_ids
original_outputs = model.generate(input_ids=input_ids,generation_config=GenerationConfig(max_new_tokens=200,num_beams=1))
original_text_output = tokenizer.decode(original_outputs[0],skip_special_tokens=True)
instruct_outputs = instruct_model.generate(input_ids=input_ids,generation_config=GenerationConfig(max_new_tokens=200,num_beams=1))
instruct_text_output = tokenizer.decode(instruct_outputs[0],skip_special_tokens=True)
dash_line = '-'.join('' for x in  range (100))
print(dash_line)
print(f'Baseline human summary is : \n{human_baseline_summary}')
print(dash_line)
print(f'Original Model generation(zero_shot) is : \n{original_text_output}')
print(dash_line)
print(f'Instruct Model generation(Fine Tuning) is :\n{instruct_text_output}')