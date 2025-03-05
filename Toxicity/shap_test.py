from shap_utils import *
from utils import *
import json


models=['gpt-3.5-turbo','glm-3-turbo','vicuna-v1.5-7b']

model=models[1]

def calculate_shap(data):
    completions = []
    for item in data:
        for c in item['res']:
            completions.append(c['completion'])

    dangerous_words = get_dangerous_words(completions)
    torch.cuda.empty_cache()

    idx = 0
    for item in data:
        for c in item['res']:
            c['dangerous_words'] = dangerous_words[idx]
            idx += 1

    return data
    


if __name__ == '__main__':
    with open(f'./results/{model}/Dangerous_words.json', 'w', encoding='utf-8') as file:
        pass

    with open(f'./results/{model}/Basic.json', 'r', encoding='utf-8') as file:
        file_content = f"[{file.read()[:-1]}]"
        data = json.loads(file_content)

        data=data

        split_data = np.array_split(data, 20)

        # split_samples = [(d,) for d in split_data]

        multiprocessing.set_start_method('spawn')

        data = run_in_parallel_with_result(calculate_shap, split_data, num_processes=len(split_data))

        data = [item for sublist in data for item in sublist]

        with open(f'./results/{model}/Dangerous_words.json', 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)