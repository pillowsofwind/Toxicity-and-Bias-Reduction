from methods import *
import json

data_type="Polarity"
models=['gpt-4o-2024-05-13']
bls=['Critic','Imagine_self','Imagine_others','Prehoc','Self']
times={'Toxicity':25,'Polarity':10}

model=models[0]

TOTAL_SAMPLES = 500

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
                'domain':item['domain'],
                'category':item['category'],
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
            file_content = f"[{f.read()[:-1]}]"
        basics=json.loads(file_content)

        for bl in bls:
            file_path = f'./results/{model}/{bl}.json'
            # 检查文件是否存在
            if not os.path.exists(file_path):
                # 如果文件不存在，创建一个空文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    pass
            with open(file_path, 'r', encoding='utf-8') as f:
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
        
            split_samples = np.array_split(data, cpu_count()-5)
            args_list = [[s, bl] for s in split_samples]
            run_in_parallel(test, args_list)