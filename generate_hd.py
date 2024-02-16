import os

# --------------------------------------------- #
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
model_type = "7b"
model_family = "llamabase"
result_path = f"./auto-labeled/output/{model_family}{model_type}"
# --------------------------------------------- #


print(f"{model_family}{model_type}")

import torch
from utils.model import get_model
from utils.gen import chat_change_with_answer

model, tokenizer, generation_config, at_id = get_model(model_type, model_family, 1)




from tqdm import tqdm
import json

import json
import torch
from tqdm import tqdm


    
def prompt_chat(title):
    return [{"role": "user", "content": f"Question: Tell me something about {title}.\nAnswer: "}]


def get_tokenized_ids(otext, title=None):
    text = otext.replace("@", "").replace("  ", " ").replace("  ", " ")
    text = tokenizer.decode(tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()[0]).replace("<s>", "").replace("</s>", "")
    if model_family == "vicuna":
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Question: Tell me something about {title}.\nAnswer: \nASSISTANT: {text}"
    if "chat" in model_family:
        return chat_change_with_answer(prompt_chat(title), text.strip(), tokenizer)
    return tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()

def get_hd(text, title=None):
    ids = get_tokenized_ids(text, title)
    hd = model(torch.tensor(ids).to(model.device), output_hidden_states=True).hidden_states
    hds = hd[1][0][-1].clone().detach()
    for i in range(2, len(hd)):
        hds += hd[i][0][-1].clone().detach()
    hds = hds / (len(hd) - 1)
    
    # only for llamachat

    if model_family == "llamachat":
        start_at = -1
        for i in range(len(ids[0])):
            if ids[0][i:i+4] == [518, 29914, 25580, 29962]:
                start_at = i
                break
        if start_at == -1:
                print("not found")
                start_at = 1
        else:
            start_at += 4
    elif model_family == "vicuna":
        ids2 = tokenizer(f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Question: Tell me something about {title}.\nAnswer: \nASSISTANT: ")['input_ids']
        start_at = -1
        ids1 = ids[0]
        for i in range(len(ids1)):
            if i >= len(ids2) or ids1[i] != ids2[i]:
                start_at = i
                break
        assert start_at != -1
    else:
        start_at = 2
    
    hds_mean_1 = torch.mean(hd[1][0][start_at-1:], dim=0)
    assert hds_mean_1.shape[0] == hd[1][0][-1].shape[-1]
    hds_mean_2 = torch.mean(hd[-1][0][start_at-1:], dim=0)

    return hds.tolist(), hds_mean_1.tolist(), hds_mean_2.tolist()


for data_type in ["train", "valid", "test"]:
    data = json.load(open(f"{result_path}/data_{data_type}.json", encoding='utf-8'))
    results_last = []
    results_mean1 = []
    results_mean2 = []

    for k in tqdm(data):
        hd_last = []
        hd_mean1 = []
        hd_mean2 = []
        
    
        hdl_origin, hdm1_origin, hdm2_origin = get_hd(k["original_text"], k["title"])

        for t in k["texts"]:
            hdl, hdm1, hdm2 = get_hd(t, k["title"])
            hd_last.append(hdl)
            hd_mean1.append(hdm1)
            hd_mean2.append(hdm2)

        results_last.append({
            "right": hdl_origin,
            "hallu": hd_last,
        })
        results_mean1.append({
            "right": hdm1_origin,
            "hallu": hd_mean1,
        })
        results_mean2.append({
            "right": hdm2_origin,
            "hallu": hd_mean2,
        })
    with open(f"{result_path}/last_token_mean_{data_type}.json", "w+") as f:
        json.dump(results_last, f)
    with open(f"{result_path}/last_mean_{data_type}.json", "w+") as f:
        json.dump(results_mean2, f)
            

