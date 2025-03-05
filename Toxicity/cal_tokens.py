path = './results/glm-3-turbo/'

from prompt import *

type_adj="biased"
type_n="polarity" 


from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def cal_tokens(text):
    tokens = tokenizer.tokenize(text)
    return len(tokens)

import os
import json

from tqdm import tqdm

file_names = ['Self.json','Basic.json', 'Prehoc.json', 'Critic.json', 'Shap.json','Roles.json', 'Imagine_others.json', 'Imagine_self.json']
# file_names = ['Imagine_self.json', 'Imagine_others.json']

json_files = [f for f in os.listdir(path) if f.endswith('.json')]

for json_file in json_files:

    if json_file not in file_names:
        continue
    

    file_path = f'{path}{json_file}'
    print(file_path)

    final_data=[]

    with open(file_path, 'r') as file:
        file_content = file.read()
    
    data = json.loads(file_content)

    num_inputs = 0
    num_outputs = 0
    for item in tqdm(data):
        for comp in item["res"]:
            inputs=''
            outputs=''
            if json_file == 'Basic.json':
                sample = item["prompt"]
                inputs += INIT_PROMPT
                inputs += BA_REQUIREMENT_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_CORRECTION_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj, type_n=type_n)
                outputs += comp['completion']
            elif json_file == 'Prehoc.json':
                sample = item["prompt"]
                inputs += INIT_PROMPT
                inputs += PH_PROMPT
                inputs += BA_REQUIREMENT_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_CORRECTION_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj, type_n=type_n)
                outputs += comp['completion']
            elif json_file == 'Self.json':
                sample = item["prompt"]
                inputs += INIT_PROMPT
                inputs += BA_REQUIREMENT_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_CORRECTION_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj, type_n=type_n)
                outputs += comp['output']['orig']
                inputs += inputs
                inputs += comp['output']['orig']
                inputs += SC_REVIEW_PROMPT.format(type_adj=type_adj,type_n=type_n)
                outputs += comp['output']['review']
                inputs += inputs
                inputs += comp['output']['review']
                inputs += SC_CORRECT_PROMPT.format(type_adj=type_adj,type_n=type_n)
                outputs += comp['completion']
            elif json_file == 'Critic.json':
                sample = item['prompt']
                inputs += INIT_PROMPT
                inputs += BA_REQUIREMENT_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_CORRECTION_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj, type_n=type_n)
                outputs += comp['output']['orig']
                num_try = len(comp['output']['retry'])
                last_response = comp['output']['orig']
                for idx in range(num_try):
                    inputs += inputs
                    inputs += last_response
                    inputs += CR_REVIEW_PROMPT.format(score=str(comp['output']['retry'][idx]['maxscore'] * 100)[:4] + '%',attr=comp['output']['retry'][idx]['maxattr'],type_adj=type_adj,type_n=type_n)
                    outputs += comp['output']['retry'][idx]['response']
                    last_response = comp['output']['retry'][idx]['response']
            elif json_file == 'Shap.json':
                sample = item['prompt']
                inputs += INIT_PROMPT
                inputs += BA_REQUIREMENT_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_CORRECTION_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj, type_n=type_n)
                outputs += comp['output']['orig']
                dangerous_words = comp['output']['dangerous_words']
                inputs += inputs
                inputs += ("\n").join(dangerous_words)
                inputs += SH_REVIEW_PROMPT.format(type_adj=type_adj,type_n=type_n)
                outputs += comp['output']['review']
                inputs += inputs
                inputs += comp['output']['review']
                inputs += SH_CORRECT_PROMPT.format(type_adj=type_adj,type_n=type_n)
                outputs += comp['completion']
            elif json_file == 'Roles.json':
                sample = item['prompt']
                inputs += INIT_PROMPT
                inputs += BA_REQUIREMENT_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_CORRECTION_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj, type_n=type_n)
                outputs += comp['output']['orig']
                inputs += INIT_PROMPT
                inputs += RP_GEN_PROMPT.format(type_adj=type_adj,type_n=type_n,num=3)
                for role in comp['output']['roles']:
                    outputs+=role['role']
                    outputs+=role['profession']
                    inputs += INIT_PROMPT
                    inputs += RP_ROLE_PROMPT.format(role=role['role'],profession=role['profession'],sample=sample,response_orig=comp['output']['orig'],type_adj=type_adj,type_n=type_n)
                a_str=""
                for role_resp in comp['output']['role_response']:
                    outputs += role_resp
                    a_str = a_str + role_resp + "\n"
                inputs += INIT_PROMPT
                inputs += RP_SORT_PROMPT.format(response_orig=comp['output']['orig'],str=a_str,type_adj=type_adj,type_n=type_n)
                outputs += comp['output']['sorted']
                inputs += BA_REQUIREMENT_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_CORRECTION_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj, type_n=type_n)
                inputs += INIT_PROMPT
                inputs += RP_MODIFICATION_PROMPT.format(co_str=comp['output']['sorted'],type_adj=type_adj,type_n=type_n)
                outputs += comp['completion']
            elif json_file == 'Imagine_others.json':
                sample = item["prompt"]
                inputs += INIT_PROMPT
                inputs += BA_REQUIREMENT_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_CORRECTION_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj, type_n=type_n)
                outputs += comp['output']['orig']
                inputs += SP_AUD_PROMPT_NUM.format(num=5)
                outputs += comp['output']['aud']
                inputs += inputs
                inputs += comp['output']['aud']
                inputs += SPO_IMAGING_PROMPT.format(type_adj=type_adj)
                outputs += comp['output']['feel']
                inputs += inputs
                inputs += comp['output']['feel']
                inputs += SP_CORRECT_PROMPT.format(type_adj=type_adj)
                outputs += comp['completion']
            elif json_file == 'Imagine_self.json':
                sample = item["prompt"]
                inputs += INIT_PROMPT
                inputs += BA_REQUIREMENT_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_CORRECTION_PROMPT.format(type_adj=type_adj, type_n=type_n)
                inputs += BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj, type_n=type_n)
                outputs += comp['output']['orig']
                inputs += SP_AUD_PROMPT_NUM.format(num=5)
                outputs += comp['output']['aud']
                inputs += inputs
                inputs += comp['output']['aud']
                inputs += SPS_IMAGING_PROMPT.format(type_adj=type_adj)
                outputs += comp['output']['feel']
                inputs += inputs
                inputs += comp['output']['feel']
                inputs += SP_CORRECT_PROMPT.format(type_adj=type_adj)
                outputs += comp['completion']
            num_inputs+=cal_tokens(inputs)
            num_outputs+=cal_tokens(outputs)

                

    print(num_inputs)
    print(num_outputs)
        