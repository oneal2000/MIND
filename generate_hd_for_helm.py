import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from utils.model import get_model
from utils.gen import chat_change_with_answer, get_pe
import json


from tqdm import tqdm

model_type = "7b"
model_family = "llamabase"
print(f"{model_family}{model_type}")

def prompt_chat_for_labeled(prompt):
    return [{"role": "user", "content": prompt}]
    

def get_tokenized_ids(answer, q):

    text = f"{q.strip()} {answer.strip()}"
    otext = q.strip()

    if "chat" in model_family:
        id1 = chat_change_with_answer(prompt_chat_for_labeled(q), answer.strip(), tokenizer)[0]
        start_at = -1
        for i in range(len(id1)):
            if id1[i:i+4] == [518, 29914, 25580, 29962]:
                start_at = i
        if start_at == -1:
            raise Exception
        else:
            start_at += 4
        
    else:
        id1 = tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()[0]
        id2 = tokenizer(otext.strip(), return_tensors='pt')['input_ids'].tolist()[0]
        start_at = -1
        for i in range(len(id1)):
            if i >= len(id2) or id1[i] != id2[i]:
                start_at = i
                break
    return [id1], start_at

def get_hd(answer, q):
    ids, start_at = get_tokenized_ids(answer, q)
    op = model(torch.tensor(ids).to(model.device), output_hidden_states=True)
    hd = op.hidden_states
    hds = hd[1][0][-1].clone().detach()
    for i in range(2, len(hd)):
        hds += hd[i][0][-1].clone().detach()
    hds = hds / (len(hd) - 1)
    
    hds_mean = torch.mean(hd[-1][0][start_at-1:], dim=0)
    assert hds_mean.shape[0] == hd[1][0][-1].shape[-1]
    
    logit = op.logits
    pl, el = get_pe(logit, ids[0], start_at)
    
    return hds.tolist(), hds_mean.tolist(), pl, el


model, tokenizer, generation_config, at_id = get_model(model_type, model_family, 1)


hd_result_path = f"./helm/hd/{model_family}{model_type}"
    

if not os.path.exists(hd_result_path):
    os.mkdir(hd_result_path)
with open(f"./helm/data/{model_family}{model_type}/data.json", "r", encoding='utf-8') as f:
    data = json.load(f)
result = {}
for d in tqdm(data):
    result[d] = {
        "sentences": [],
        "passage": None,
    }
    prompt = data[d]["prompt"]
    prompt = tokenizer.decode(tokenizer(prompt.strip(), return_tensors='pt')['input_ids'].tolist()[0]).replace("<s>", "").replace("</s>", "")
    for s in data[d]["sentences"]:
        hd_last, hd_last_mean, pl, el = get_hd(s["sentence"], prompt)
        result[d]["sentences"].append({
            "hd_last_token": hd_last,
            "hd_last_mean": hd_last_mean,
            "probability": pl,
            "entropy": el
        })
    hd_last, hd_last_mean, pl, el = get_hd(" ".join([t["sentence"] for t in data[d]["sentences"]]), prompt)
    result[d]["passage"] = {
        "hd_last_token": hd_last,
        "hd_last_mean": hd_last_mean,
        "probability": pl,
        "entropy": el
    }
with open(f"{hd_result_path}/hd.json", "w+") as f:
    json.dump(result, f)
