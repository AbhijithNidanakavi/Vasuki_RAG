import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch

DEVICE = 'cpu'
torch_device = torch.device(DEVICE)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2',pad_token_id=tokenizer.eos_token_id).to(torch_device)

## Text Generation using Greedy_Search ##
model_inputs = tokenizer('I am hungry because ', return_tensors = 'pt').to(torch_device)

## Generating New Tokens ##
greedy_output = model.generate(**model_inputs,max_new_tokens = 50)

print('Generated Output :')
print(tokenizer.decode(greedy_output[0],skip_special_tokens=True))

## Text Generation using Beam_Search ##
print('-'*50)
beam_output = model.generate(**model_inputs, max_new_tokens = 50, num_beams = 2,early_stopping = True)
print('Generated Output :')
print(tokenizer.decode(beam_output[0],skip_special_tokens=True))

## Text Generation using Beam_Search with n_gram ##
print('-'*50)
beam_output = model.generate(**model_inputs, max_new_tokens = 50, num_beams = 5,early_stopping = True, no_repeat_ngram_size = 2)
print('Generated Output :')
print(tokenizer.decode(beam_output[0],skip_special_tokens=True))

# ## Text Generation using Beam_Search with num_return_sequences ##
# beam_outputs = model.generate(**model_inputs, max_new_tokens = 50, num_beams = 5,early_stopping = True, no_repeat_ngram_size = 2, num_repeat_sequences = 5)
# print('Generated Output :')
# ##print(tokenizer.decode(beam_output[0],skip_special_tokens=True))
# for i,beam_output in enumerate(beam_outputs):
#     print('{}:{}'.format(i,tokenizer.decode(beam_output[0],skip_special_tokens=True)))
    
## Sampling Algorithm ## 
print('-'*50)
from transformers import set_seed
set_seed(5)
sample_output = model.generate(**model_inputs, max_new_tokens = 50, do_sample=True)
print(tokenizer.decode(sample_output[0]))

## Clipping the Sampling Algorithm ##
print('-'*50)
sample_output = model.generate(**model_inputs, max_new_tokens = 50, do_sample=True, top_k=80)
print(tokenizer.decode(sample_output[0]))

## Top-p(nucleus) sampling ##
print('-'*50)
sample_output = model.generate(**model_inputs, max_new_tokens = 50, do_sample=True,top_p=0.92,top_k=0)
print(tokenizer.decode(sample_output[0]))

## Top-p(nucleus) sampling with top-k ##
print('-'*50)
sample_outputs = model.generate(**model_inputs, max_new_tokens = 50, do_sample=True,top_p=0.95,top_k=50)
for i, sample_output in enumerate(sample_outputs):
    print('{}:{}'.format(i,tokenizer.decode(sample_output)))
