from utils import *
from llama_cpp import Llama
import os
import numpy as np
from prompt import *
from evaluates import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

llm={}
def uni_chat(model_name,messages):
    if model_name == 'llama-2-7b-chat':
        str=""
        for message in messages:
            str+=message['content']
            if len(str.split())>4096:
                return ""
        completion = llama_chat_completion_with_backoff(
            model=model_name,
            messages=messages,
            temperature=0.7,
            top_p=0.9
        )
        return completion.choices[0].message.content.strip()
    elif model_name == 'vicuna-v1.5-7b':
        str=""
        for message in messages:
            str+=message['content']
            if len(str.split())>4096:
                return ""
        completion = vicuna_chat_completion_with_backoff(
            model=model_name,
            messages=messages,
            temperature=0.7,
            top_p=0.9
        )
        return completion.choices[0].message.content.strip()
    elif model_name=='gpt-4-1106-preview' or model_name=='gpt-3.5-turbo' or model_name=='gpt-4o-mini' or model_name=='claude-3-haiku-20240307':
        # # print('debug')
        # completion = ft_chat_completion_with_backoff(
        #     model= 'ft:gpt-3.5-turbo-0613:personal::8obiVxO7',
        #     messages=messages,
        #     # temperature=0.7,
        #     # top_p=0.9
        # )
        completion = chat_completion_with_backoff(
            model=model_name,
            messages=messages,
            temperature=0.7,
            top_p=0.9
        )
        return completion.choices[0].message.content.strip()
    elif model_name=='ft:gpt-3.5-turbo-0613:personal::8obiVxO7':
        completion = ft_chat_completion_with_backoff(
            model=model_name,
            messages=messages,
            temperature=0.7,
            top_p=0.9
        )
        return completion.choices[0].message.content.strip()
    else:
        return do_request(model_name,messages)


def gen_roles(type,model_name,num=3):
    type_adj=""
    type_n=""
    if type=="Toxicity":
        type_adj="toxic"
        type_n="toxicity"
    else:
        type_adj="biased"
        type_n="polarity"    
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role":"user","content":RP_GEN_PROMPT.format(type_adj=type_adj,type_n=type_n,num=num)}
    ]
    response=uni_chat(model_name,messages)
    # print(response)
    professions=[]
    roles=[]
    i=0
    for line in response.split('\n'):
        if len(line)<=1 or line.find(':')==-1:
            continue
        if len(line.split(':')[0])<=1 or len(line.split(':')[1])<=1:
            continue
        i=i+1
        if i>num:
            break
        professions.append(line.split(':')[0])
        roles.append("[Your role]\n"+line.split(':')[0]+'\n[Your responsibility]\n'+line.split(':')[1])

    if len(roles) != num or len(professions) != num:
        return gen_roles(type,model_name,num)
    return professions,roles



def detox(data_type,method_name,model_name,sample,original_response=None,dangerous_words=None,aud_response=None,self_feel=None,other_feel=None):
    output={}
    type_adj=""
    type_n=""
    if data_type=="Toxicity":
        type_adj="toxic"
        type_n="toxicity"
    else:
        type_adj="biased"
        type_n="polarity" 

    if method_name=="Basic":
        messages = [
            {"role": "system", "content": INIT_PROMPT},
            {"role": "user", "content": BA_REQUIREMENT_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "assistant", "content": BA_CORRECTION_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "user", "content": BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj,type_n=type_n)}
        ]
        return uni_chat(model_name,messages),output
    elif method_name=="Self":
        messages = [
            {"role": "system", "content": INIT_PROMPT},
            {"role": "user", "content": BA_REQUIREMENT_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "assistant", "content": BA_CORRECTION_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "user", "content": BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj,type_n=type_n)}
        ]
        response_orig=original_response
        output['orig']=response_orig
        messages.append({"role": "assistant", "content": response_orig})
        messages.append({"role": "user", "content": SC_REVIEW_PROMPT.format(type_adj=type_adj,type_n=type_n)})
        response = uni_chat(model_name,messages)
        output['review']=response
        # print('[Review]:'+response)
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": SC_CORRECT_PROMPT.format(type_adj=type_adj,type_n=type_n)})
        return uni_chat(model_name,messages),output
    elif method_name=="Prehoc":
        messages = [
            {"role": "system", "content": PH_PROMPT+INIT_PROMPT},
            {"role": "user", "content": BA_REQUIREMENT_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "assistant", "content": BA_CORRECTION_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "user", "content": BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj,type_n=type_n)}
        ]
        return uni_chat(model_name,messages),output
    elif method_name=="Critic":
        messages = [
            {"role": "system", "content": INIT_PROMPT},
            {"role": "user", "content": BA_REQUIREMENT_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "assistant", "content": BA_CORRECTION_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "user", "content": BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj,type_n=type_n)}
        ]
        response=original_response
        output['orig']=response
        # print('[Orig.]:'+response)
        score,attr=get_max_score(response)
        output['maxscore']=score
        output['maxattr']=attr
        output['retry']=[]
        # print(score,attr)
        for i in range(4):
            prompt=CR_REVIEW_PROMPT.format(score=str(score * 100)[:4] + '%',attr=attr,type_adj=type_adj,type_n=type_n)
            # print('[Prompt]:'+prompt)
            n_message=messages
            n_message.append({"role":"assistant","content":response})
            n_message.append({"role":"user","content":prompt})
            response=uni_chat(model_name,n_message)
            # print('[Modified v{i}]:'+response)
            score,attr=get_max_score(response)
            # print(score,attr)
            retry_i={'time':i,'response':response,'maxscore':score,'maxattr':attr}
            output['retry'].append(retry_i)
            if score<0.1:
                break
        return response,output
    elif method_name=="Shap":
        messages = [
            {"role": "system", "content": INIT_PROMPT},
            {"role": "user", "content": BA_REQUIREMENT_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "assistant", "content": BA_CORRECTION_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "user", "content": BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj,type_n=type_n)}
        ]
        response_orig=original_response
        output['orig']=response_orig
        # print('[Orig.]:'+response_orig)
        dwords=dangerous_words
        output['dangerous_words']=dwords
        prompt=SH_REVIEW_PROMPT.format(type_adj=type_adj,type_n=type_n)+'\n'
        for word in dwords:
            prompt+=word+'\n'
        # print('[Prompt]:'+prompt)
        messages.append({"role": "assistant", "content": response_orig})
        messages.append({"role": "user", "content": prompt})
        response = uni_chat(model_name,messages)
        output['review']=response
        # print('[Review]:'+response)
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": SH_CORRECT_PROMPT.format(type_adj=type_adj,type_n=type_n)})
        return uni_chat(model_name,messages),output
    elif method_name=="Roles":
        professions,roles=gen_roles(data_type,model_name)
        role_addr='./'+'results'+'/'+model_name+'/'
        role_output_file_name=role_addr+'all_roles.json'
        if not os.path.exists(role_addr):
                os.makedirs(role_addr)
        with open(role_output_file_name, 'a' ) as f:
            f.write(json.dumps(roles,ensure_ascii=False,indent=4))
        messages = [
            {"role": "system", "content": INIT_PROMPT},
            {"role": "user", "content": BA_REQUIREMENT_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "assistant", "content": BA_CORRECTION_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "user", "content": BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj,type_n=type_n)}
        ]
        response_orig=original_response
        output['orig']=response_orig
        output['roles']=[]
        # print('[Orig.]:'+response_orig)
        resps=[]
        ni=0
        for role in roles:
            profession=professions[ni]
            output['roles'].append({"role": role, "profession": profession})
            ni+=1
            prompt=RP_ROLE_PROMPT.format(role=role,profession=profession,sample=sample,response_orig=response_orig,type_adj=type_adj,type_n=type_n)
            messages=[
                {"role": "system", "content": INIT_PROMPT},
                {"role":"user","content":prompt}
            ]
            response=uni_chat(model_name,messages)
            # print('[Resp.]:'+response)
            resps.append(response)
        output['role_response']=resps
        a_str=""
        for resp in resps:
            a_str=a_str+resp+'\n'
        prompt=RP_SORT_PROMPT.format(response_orig=response_orig,str=a_str,type_adj=type_adj,type_n=type_n)
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role":"user","content":prompt}
            ]
        co_str=uni_chat(model_name,messages)
        output['sorted']=co_str
        # print('[Sorted]:'+co_str)
        messages = [
            {"role": "system", "content": INIT_PROMPT},
            {"role": "user", "content": BA_REQUIREMENT_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "assistant", "content": BA_CORRECTION_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "user", "content": BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj,type_n=type_n)},
            {"role":"assistant","content":response_orig},
            {"role":"user","content": RP_MODIFICATION_PROMPT.format(co_str=co_str,type_adj=type_adj,type_n=type_n)}
        ]
        output['final_prompt']=RP_MODIFICATION_PROMPT.format(co_str=co_str,type_adj=type_adj,type_n=type_n)
        return uni_chat(model_name,messages),output
    elif method_name=='Contrast':
        professions,roles=gen_roles(data_type,model_name,1)
        role=roles[0]
        profession=professions[0]
        output['role']={"role": role, "profession": profession}
        messages0 = [
            {"role": "system", "content": INIT_PROMPT},
            {"role": "user", "content": PH_PROMPT+BA_REQUIREMENT_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "assistant", "content": BA_CORRECTION_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "user", "content": BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj,type_n=type_n)}
        ]
        response_orig=uni_chat(model_name,messages0)
        # print('[Orig.]:'+response_orig)
        output['orig']=response_orig
        messages = [
            {"role": "system", "content": INIT_PROMPT},
            {"role": "user", "content": CT_BAD_PROMPT.format(type_adj=type_adj,type_n=type_n,sample=sample)}
        ]
        response_bad=uni_chat(model_name,messages)
        # print('[bad]:'+response_bad)
        output['bad']=response_bad
        messages = [
            {"role": "system", "content": INIT_PROMPT},
            {"role": "user", "content": CT_ROLE_PROMPT.format(role=role,sample=sample,response_orig=response_orig,response_bad=response_bad)}
        ]
        response=uni_chat(model_name,messages)
        # print('[resp.]'+response)
        output['resp']=response
        messages0.append({"role": "assistant", "content": response_orig})
        messages0.append({'role':'user','content':CT_CORRECT2_PROMPT.format(response_bad=response_bad,eva=response)})
        return uni_chat(model_name,messages0),output
    elif method_name=="Imagine_others":
        messages = [
            {"role": "system", "content": INIT_PROMPT},
            {"role": "user", "content": PH_PROMPT+BA_REQUIREMENT_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "assistant", "content": BA_CORRECTION_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "user", "content": BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj,type_n=type_n)}
        ]
        response_orig=uni_chat(model_name,messages)
        # print('[Orig.]:'+response_orig)
        output['orig']=response_orig
        messages.append({"role": "assistant", "content": response_orig})
        messages.append({"role": "user", "content": SP_AUD_PROMPT})
        response=uni_chat(model_name,messages)
        output['aud']=response
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": SPO_IMAGING_PROMPT.format(type_adj=type_adj)})
        response=uni_chat(model_name,messages)
        output['feel']=response
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": SP_CORRECT_PROMPT.format(type_adj=type_adj)})
        return uni_chat(model_name,messages),output
    elif method_name.startswith("Imagine_others"):
        import re
        match= re.search(r'\d+', method_name)
        num = int(match.group())
        messages = [
            {"role": "system", "content": INIT_PROMPT},
            {"role": "user", "content": PH_PROMPT+BA_REQUIREMENT_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "assistant", "content": BA_CORRECTION_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "user", "content": BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj,type_n=type_n)}
        ]
        response_orig=original_response
        # print('[Orig.]:'+response_orig)
        output['orig']=response_orig
        messages.append({"role": "assistant", "content": response_orig})
        messages.append({"role": "user", "content": SP_AUD_PROMPT_NUM.format(num=num)})
        response=uni_chat(model_name,messages)
        output['aud']=response
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": SPO_IMAGING_PROMPT.format(type_adj=type_adj)})
        response=uni_chat(model_name,messages)
        output['feel']=response
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": SP_CORRECT_PROMPT.format(type_adj=type_adj)})
        return uni_chat(model_name,messages),output
    elif method_name.startswith("E_Imagine_others"):
        import re
        match= re.search(r'\d+', method_name)
        ex_n = int(match.group())
        num=3
        messages = [
            {"role": "system", "content": INIT_PROMPT},
            {"role": "user", "content": PH_PROMPT+BA_REQUIREMENT_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "assistant", "content": BA_CORRECTION_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "user", "content": BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj,type_n=type_n)}
        ]
        response_orig=original_response
        output['orig']=response_orig
        messages.append({"role": "assistant", "content": response_orig})
        if ex_n==1:
            messages.append({"role": "user", "content": E_SP_AUD_PROMPT_NUM.format(num=num)})
        else:
            messages.append({"role": "user", "content": SP_AUD_PROMPT_NUM.format(num=num)})
        response=uni_chat(model_name,messages)
        output['aud']=response
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": E_SPO_IMAGING_PROMPT[ex_n].format(type_adj=type_adj)})
        response=uni_chat(model_name,messages)
        output['feel']=response
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": SP_CORRECT_PROMPT.format(type_adj=type_adj)})
        return uni_chat(model_name,messages),output
    elif method_name=="Imagine_self":
        messages = [
            {"role": "system", "content": INIT_PROMPT},
            {"role": "user", "content": PH_PROMPT+BA_REQUIREMENT_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "assistant", "content": BA_CORRECTION_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "user", "content": BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj,type_n=type_n)}
        ]
        response_orig=uni_chat(model_name,messages)
        # print('[Orig.]:'+response_orig)
        output['orig']=response_orig
        messages.append({"role": "assistant", "content": response_orig})
        messages.append({"role": "user", "content": SP_AUD_PROMPT})
        response=uni_chat(model_name,messages)
        output['aud']=response
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": SPS_IMAGING_PROMPT.format(type_adj=type_adj)})
        response=uni_chat(model_name,messages)
        output['feel']=response
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": SP_CORRECT_PROMPT.format(type_adj=type_adj)})
        return uni_chat(model_name,messages),output
    elif method_name.startswith("Imagine_self"):
        import re
        match= re.search(r'\d+', method_name)
        num = int(match.group())
        print(f'Number of audience: {num}')
        messages = [
            {"role": "system", "content": INIT_PROMPT},
            {"role": "user", "content": PH_PROMPT+BA_REQUIREMENT_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "assistant", "content": BA_CORRECTION_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "user", "content": BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj,type_n=type_n)}
        ]
        response_orig=original_response
        # print('[Orig.]:'+response_orig)
        output['orig']=response_orig
        messages.append({"role": "assistant", "content": response_orig})
        messages.append({"role": "user", "content": SP_AUD_PROMPT_NUM.format(num=num)})
        response=uni_chat(model_name,messages)
        output['aud']=response
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": SPS_IMAGING_PROMPT.format(type_adj=type_adj)})
        response=uni_chat(model_name,messages)
        output['feel']=response
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": SP_CORRECT_PROMPT.format(type_adj=type_adj)})
        return uni_chat(model_name,messages),output
    elif method_name.startswith("E_Imagine_self"):
        import re
        match= re.search(r'\d+', method_name)
        ex_n = int(match.group())
        num=3
        messages = [
            {"role": "system", "content": INIT_PROMPT},
            {"role": "user", "content": PH_PROMPT+BA_REQUIREMENT_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "assistant", "content": BA_CORRECTION_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "user", "content": BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj,type_n=type_n)}
        ]
        response_orig=original_response
        output['orig']=response_orig
        messages.append({"role": "assistant", "content": response_orig})
        if ex_n==1:
            messages.append({"role": "user", "content": E_SP_AUD_PROMPT_NUM.format(num=num)})
        else:
            messages.append({"role": "user", "content": SP_AUD_PROMPT_NUM.format(num=num)})
        # print('test1')
        response=uni_chat(model_name,messages)
        # print('test2')
        output['aud']=response
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": E_SPS_IMAGING_PROMPT[ex_n].format(type_adj=type_adj)})
        response=uni_chat(model_name,messages)
        output['feel']=response
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": SP_CORRECT_PROMPT.format(type_adj=type_adj)})
        return uni_chat(model_name,messages),output
    elif method_name=="Self_FT":
        messages = [
            {"role": "system", "content": INIT_PROMPT_FT},
            {"role": "user", "content": BA_REQUIREMENT_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "assistant", "content": BA_CORRECTION_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "user", "content": BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj,type_n=type_n)}
        ]
        response_orig=original_response
        output['orig']=response_orig
        # print('[Orig.]:'+response_orig)
        messages.append({"role": "assistant", "content": response_orig})
        messages.append({"role": "user", "content": SC_REVIEW_PROMPT_FT.format(type_adj=type_adj,type_n=type_n)})
        return uni_chat(model_name,messages),output
    elif method_name=="Imagine_combine":
        messages = [
            {"role": "system", "content": INIT_PROMPT},
            {"role": "user", "content": PH_PROMPT+BA_REQUIREMENT_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "assistant", "content": BA_CORRECTION_PROMPT.format(type_adj=type_adj,type_n=type_n)},
            {"role": "user", "content": BA_COMPLETION_PROMPT.format(sample=sample,type_adj=type_adj,type_n=type_n)}
        ]
        if original_response!=None:
            response_orig=original_response
        else:
            response_orig=uni_chat(model_name,messages)
        # print('[Orig.]:'+response_orig)
        output['orig']=response_orig
        messages.append({"role": "assistant", "content": response_orig})
        messages.append({"role": "user", "content": SP_AUD_PROMPT})
        if aud_response!=None:
            response=aud_response
        else:
            response=uni_chat(model_name,messages)
        output['aud']=response
        messages.append({"role": "assistant", "content": response})
        messages1=messages
        messages.append({"role": "user", "content": SPS_IMAGING_PROMPT.format(type_adj=type_adj)})
        messages1.append({"role": "user", "content": SPO_IMAGING_PROMPT.format(type_adj=type_adj)})
        if self_feel!=None:
            response=self_feel
        else:
            response=uni_chat(model_name,messages)
        if other_feel!=None:
            response1=other_feel
        else:
            response1=uni_chat(model_name,messages1)
        output['self_feel']=response
        output['other_feel']=response1
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": SPO_IMAGING_PROMPT.format(type_adj=type_adj)})
        messages.append({"role": "assistant", "content": response1})
        messages.append({"role": "user", "content": SP_CORRECT_PROMPT.format(type_adj=type_adj)})
        return uni_chat(model_name,messages),output