import shap
import transformers
import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# tokenizer = transformers.AutoTokenizer.from_pretrained("sasha/regardv3")
# r4_model = transformers.AutoModelForSequenceClassification.from_pretrained("sasha/regardv3").cuda()
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
r4_model = transformers.AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target").cuda()


def predict_function(texts):
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
        texts = [str(text) for text in texts] 

    # Now, texts is guaranteed to be a list, which the tokenizer can process
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(r4_model.device)
    with torch.no_grad():
        outputs = r4_model(**inputs)
    return outputs.logits.cpu().numpy()  # Ensure the output is in a numpy array format for SHAP
explainer = shap.Explainer(predict_function, tokenizer)

# def get_dangerous_words(text):
#     shap_values=explainer(text)
#     nans=[]
#     for idx in range(len(text)):
#         words=[]
#         for i, word in enumerate(shap_values.data[idx]):
#             # the second score is toxicity score
#             score = shap_values.values[idx][i][1]
#             if score>0.5:
#                 words.append(word)
#         nans.append(words)
#     return nans

def get_dangerous_words(text):
    shap_values = explainer(text)
    top_words = [] 
    for idx in range(len(text)):
        word_scores = [] 
        for i, word in enumerate(shap_values.data[idx]):
            score = shap_values.values[idx][i][1]
            word_scores.append((word, score))

        top_two_words = sorted(word_scores, key=lambda x: x[1], reverse=True)[:2]
        top_words.append([word for word, _ in top_two_words])

    return top_words

# print(get_dangerous_words(["you are a piece of shit, I gonna kill ya you little piece of shit","what the hell fuck", "moron guy she is"]))