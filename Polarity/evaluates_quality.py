from googleapiclient import discovery
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) # for exponential backoff
import evaluate
import numpy as np
import os
from utils import *
from tqdm.auto import tqdm
from transformers import logging
logging.set_verbosity_error()



from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def ppl_compute(text):
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    from evaluate import load
    perplexity = load("perplexity",  module_type= "measurement")
    results=perplexity.compute(data=text, model_id='gpt2-large',batch_size=32)
    return results['mean_perplexity']


def perplexity_compute(data,batch_size=16):
    ppl = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        inputs = tokenizer_gpt2(batch, padding=True, truncation=True, return_tensors="pt").to('cuda')

        with torch.no_grad():
            outputs = model_gpt2(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()  
            ppl.append(perplexity/len(batch))

    return ppl

from bert_score import score

def bertscore_compute(inputs,references,batch_size=64):
    scores = []

    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        batch_references = references[i:i+batch_size]

        P, R, F1 = score(batch_inputs, batch_references, lang="en", verbose=False, device='cuda', model_type="distilbert-base-uncased", batch_size=batch_size)

        scores.extend(F1.tolist())

    return scores
   
import math

def HF_L1(s1,s2):
    return abs(s1['positive']-s2['positive'])+abs(s1['neutral']-s2['neutral'])+abs(s1['negative']-s2['negative'])
def HF_L2(s1,s2):
    return math.sqrt(math.pow(s1['positive']-s2['positive'],2)+math.pow(s1['neutral']-s2['neutral'],2)+math.pow(s1['negative']-s2['negative'],2))
def Sent_L1(s1,s2):
    return abs(s1['neg']-s2['neg'])+abs(s1['neu']-s2['neu'])+abs(s1['pos']-s2['pos'])
def L1(metric,s1,s2):
    if metric=='HF':
        return HF_L1(s1,s2)
    else:
        return Sent_L1(s1,s2)
# metric: HF(HF Regard),Sent(VADER)

    
import re
def cut_off_completion(text):
    s=re.search('Completion(.*)',text)
    if s==None:
        s=re.search('completion(.*)',text)
        if s==None:
            return text
        else:
            return s.group(1)
    else:
        return s.group(1)

import re
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

def get_ngram(text, n):
    tokens = word_tokenize(text)
    return list(ngrams(tokens, n))

def diversity(text,n):

    unique_ngrams = len(set(get_ngram(text,n)))
    
    tokens = word_tokenize(text)
    sentence_length = len(tokens)
    
    if sentence_length == 0:  # 防止除以零
        return 0
    diversity = unique_ngrams / sentence_length
    return diversity

def diversity_compute(group):
    nans=[]
    for text in group:
        nt=[]
        for i in range(1,4):
            nt.append(diversity(text,i))
        nans.append(nt)
    return nans

def get_diversity_score(group):
    split_group=np.array_split(group,cpu_count()-5)
    data = run_in_parallel_with_result(diversity_compute,split_group,num_processes=len(split_group))
    data = [item for sublist in data for item in sublist]
    return data


import csv

# ppl = perplexity_compute(["Hello, my name is Donald Trump, I am the president of USA."])

# print(ppl)


# data_type="Polarity"
# models=['gpt-3.5-turbo','glm-3-turbo', 'llama-2-7b-chat']
# bls=['Basic','Self','Prehoc','Critic','Shap','Roles','Contrast','Imagine_self','Imagine_others']
# # bls=['Basic']

# model=models[0]

# if __name__ == '__main__':
#     all_scores=[]
#     all_scores.append(['method','PPL','BERTScore','Q'])
#     for bl in bls:
#         with open(f'./results/{model}/{bl}.json', 'r', encoding='utf-8') as f:
#             file_content = f.read()
#             output=json.loads(file_content)
#         with open(f'./results/{model}/Basic.json', 'r', encoding='utf-8') as f:
#             basic_file_content = f.read()
#             basic_output=json.loads(basic_file_content)
#         category=["European_Americans", "Asian_Americans", "African_Americans","American_actresses","American_actors"]

#         completions=[]
#         prompts=[]
#         basic_comps=[]
#         for item,basic_item in list(zip(output,basic_output)):
#             if item['category'] not in category:
#                 continue
#             prompt=item["prompt"]
#             for t in item['res']:
#                 completion=cut_off_completion(t["completion"])
#                 completions.append(completion)
#                 prompts.append(prompt)
#             for t in basic_item['res']:
#                 completion=cut_off_completion(t["completion"])
#                 basic_comps.append(completion)
        

#         ppls = perplexity_compute(completions)

#         print(f'PPL ({bl}) = {np.mean(ppls)}')
#         bert_scores = bertscore_compute(completions, basic_comps)

    #     avg_ppl = np.mean(ppls)
    #     avg_bert_score = np.mean(bert_scores)

    #     avg_bert_score_uniform = (avg_bert_score+1)/2 * 100

    #     quality = np.sqrt(np.mean(np.square([100-avg_ppl,avg_bert_score_uniform])))

    #     print(f'Average perplexity: {avg_ppl}')
    #     print(f'Average BERTScore: {avg_bert_score}')
    #     print(f'Average BERTScore (Uniformed): {avg_bert_score_uniform}')
    #     print(f'Quality: {quality}')
            
            
   
        
    #     all_scores.append([bl,avg_ppl,avg_bert_score,quality])
        
       
    # addr=f'./results/{model}'
    # if not os.path.exists(addr):
    #     os.makedirs(addr)
    # output_file_name=f'{addr}/scores_quality.csv'
    # with open(output_file_name, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(all_scores)
