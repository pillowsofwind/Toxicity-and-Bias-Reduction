from openai import OpenAI       
# from zhipuai import ZhipuAI
import json
import requests 


client_openai = OpenAI(
    api_key = ''
)

client_llama = OpenAI(
    base_url = "",
    api_key = ''
)

client_vicuna = OpenAI(
    base_url = "",
    api_key = ''
)

client_ft = OpenAI(
    api_key = ''
)


# set ZhipuAI API key
ZhipuAI_API_KEY = ''

# set perspective API key
dev_API_KEY = ''

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) # for exponential backoff

## wait to avoid the limit
@retry(wait=wait_random_exponential(min=30, max=120), stop=stop_after_attempt(1000))
def chat_completion_with_backoff(**kwargs):
    return client_openai.chat.completions.create(**kwargs)

@retry(wait=wait_random_exponential(min=30, max=120), stop=stop_after_attempt(10))
def llama_chat_completion_with_backoff(**kwargs):
    return client_llama.chat.completions.create(**kwargs)

@retry(wait=wait_random_exponential(min=30, max=120), stop=stop_after_attempt(10))
def vicuna_chat_completion_with_backoff(**kwargs):
    return client_vicuna.chat.completions.create(**kwargs)

@retry(wait=wait_random_exponential(min=30, max=120), stop=stop_after_attempt(1000))
def ft_chat_completion_with_backoff(**kwargs):
    return client_ft.chat.completions.create(**kwargs)



## wait to avoid the limit
@retry(wait=wait_random_exponential(min=30, max=120), stop=stop_after_attempt(1000))
def completion_with_backoff(**kwargs):
    return client_openai.completions.create(**kwargs)



def conversation_to_string(conversation_history):
    idx = 0
    if len(conversation_history) < 1:
        return None
    # check if there's system prompt
    if conversation_history[0]['role'] == "system":
        system = conversation_history[0]['content']
        system = f"{system}\n\n"
        idx += 1
    else:
        system = ""
        
    prompt = system
        
    while idx < len(conversation_history):
        if conversation_history[idx]['role'].lower() == 'user':
            prompt += f"user: {conversation_history[idx]['content']}\n"
        elif conversation_history[idx]['role'].lower() == 'assistant':
            prompt += f"assistant: {conversation_history[idx]['content']}\n"
        idx += 1
    
    prompt += "assistant:"
    return prompt


from multiprocessing import Process, cpu_count

# 并行
def run_in_parallel(target_function, args_list):
    processes = []

    # 创建进程
    for args in args_list:
        proc = Process(target=target_function, args=(args,))
        processes.append(proc)
        proc.start()

    # 等待所有进程完成
    for proc in processes:
        proc.join()


import multiprocessing

def run_in_parallel_with_result(target_function, args_list, num_processes=None):
    # 使用进程池，指定进程数量
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用map方法并行执行target_function
        results = pool.map(target_function, args_list)
    return results


@retry(wait=wait_random_exponential(min=30, max=120), stop=stop_after_attempt(1000))
def do_request(model_name, messages):
    client = ZhipuAI(api_key=ZhipuAI_API_KEY)
    response = client.chat.completions.create(
        model="glm-3-turbo",
        messages=messages,
        temperature=0.7,
        top_p=0.9
        )
    return response.choices[0].message.content

from googleapiclient import discovery    


req_items=['TOXICITY','SEVERE_TOXICITY','IDENTITY_ATTACK','INSULT','PROFANITY','THREAT']
@retry(wait=wait_random_exponential(min=30, max=120), stop=stop_after_attempt(1000))
def pers_call(text):
    max_len=20000
    if len(text)>max_len:
        text=text[:max_len-1]
    if len(text)==0:
        text=" "
    client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=dev_API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
    )
    req_attr={}
    for item in req_items:
        req_attr[item]={}
    analyze_request = {
    'comment': { 'text': text },
    'requestedAttributes': req_attr,
    'languages': ['en'],
    }

    response = client.comments().analyze(body=analyze_request).execute()
    return response['attributeScores']
