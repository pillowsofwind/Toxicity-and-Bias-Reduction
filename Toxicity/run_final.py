from methods import *
import json

data_type="Toxicity"
models=['gpt-3.5-turbo','glm-3-turbo','vicuna-v1.5-7b','gpt-4-1106-preview']
bls=['Imagine_self','Imagine_others']
# bls=['Critic']
times={'Toxicity':25,'Polarity':10}

model=models[1]

TOTAL_SAMPLES = 1604

basics=[]
o_output={}

def test(data):
    bl=data[1]
    data=data[0]
    addr=f'./results/{model}'
    output_file_name=f'{addr}/{bl}.json'
    for item in data:
        sample=item['prompt']
        rep_time=times[data_type]
        res=[]
        for i in range(rep_time):
            original_response=item['res'][i]['completion']
            response,output=detox(data_type,bl,model,sample,original_response=original_response)
            res.append({'completion':response,'output':output})
        result={'id':item['id'],
                # 'domain':item['domain'],
                # 'category':item['category'],
                'prompt':item['prompt'],
                'res':res}
        if not os.path.exists(addr):
                os.makedirs(addr)
        with open(output_file_name, 'a' ) as f:
            f.write(json.dumps(result,ensure_ascii=False,indent=4)+',')



if __name__ == '__main__':
    print(f'Model: {model}')
    if True:
        with open(f'./results/{model}/Basic.json', 'r', encoding='utf-8') as f:
            # file_content = f"[{f.read()[:-1]}]"
            file_content = f.read()
        basics=json.loads(file_content)

        for bl in bls:
            with open(f'./results/{model}/{bl}.json', 'r', encoding='utf-8') as f:
                file_content = f"[{f.read()[:-1]}]"
            # check if the bl is ok
            data=json.loads(file_content)

            id_set = set()
            for d in data:
                id_set.add(d['id'])

            num_left = TOTAL_SAMPLES -len(id_set)
            print(f'{bl}, left samples: {num_left}')

            if num_left > 0:
                o_output[bl] = []
                for item in basics:
                    if item['id'] in id_set:
                        pass
                    else:
                        o_output[bl].append(item)
        
        print("======RUN EXPERIMENT======")
        for bl in o_output:
            data = o_output[bl]
            print(bl,len(data))

            # print(data)

            # test([data,bl])

            
        
            split_samples = np.array_split(data, 1)
            args_list = [[s, bl] for s in split_samples]
            run_in_parallel(test, args_list)