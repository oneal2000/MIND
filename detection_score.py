import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.nn.functional as F
import json
from sklearn.metrics import precision_recall_curve, auc
import numpy as np

task_name = "helm"

def get_AUC(preds, human_labels, pos_label=1, oneminus_pred=False):
    
    preds = [v for v in preds]
    assert len(preds) == len(human_labels)
    P, R, thre = precision_recall_curve(human_labels, preds, pos_label=pos_label)
    return auc(R, P) * 100


class Model():
    def __init__(self, input_size, path):
        
        self.model = nn.Sequential()
        self.model.add_module("dropout", nn.Dropout(0.2))
        self.model.add_module(f"linear1", nn.Linear(input_size, 256))
        self.model.add_module(f"relu1", nn.ReLU())
        self.model.add_module(f"linear2", nn.Linear(256, 128))
        self.model.add_module(f"relu2", nn.ReLU())
        self.model.add_module(f"linear3", nn.Linear(128, 64))
        self.model.add_module(f"relu3", nn.ReLU())
        self.model.add_module(f"linear4", nn.Linear(64, 2))
        self.device = "cuda:0"
        self.model.load_state_dict(torch.load(path, map_location = "cpu")["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
    def eval(self, hd):
        # assert len(llama[0]) == 4096*2
        input_ = torch.tensor([hd]).to(self.device)
        score = self.model(input_)
        hallu_sm = F.softmax(score, dim=1)[:, 1]

        return hallu_sm[0].item()



root_path = f"./{task_name}"
model = os.listdir(root_path + "/hd")
model = sorted(model)

result_sent_halu = {"Our_score": {}}
result_psg_corr = {"Our_score": {}}
result_psg_halu = {"Our_score": {}}
result_sent_corr = {"Our_score": {}}

for mo in tqdm(model):
    ckpt_path = f"./auto-labeled/output/{mo}/train_log/best_acc_model.pt"
    input_size = (4096*2 if "falcon" not in mo else 4544*2) if "7b" in mo else (5120*2 if "13b" in mo else 8192*2)
    mlp = Model(input_size, ckpt_path)


    if task_name == "helm":
        hd_result_path = f"{root_path}/hd/{mo}/hd.json"
        labeled = f"{root_path}/data/{mo}/data.json"

        with open(hd_result_path) as f:
            hd = json.load(f)
        with open(labeled) as f:
            labeled = json.load(f)
        labels = []
        pre = []
        psglabels = []
        psgpre = []
        psglabelsbysent = []
        for k in labeled:
            dts = labeled[k]["sentences"]
            hds = hd[k]["sentences"]
            psg_bi = 0
            psg_not_bi = 0
            for dt, d in zip(dts, hds):
                score = mlp.eval(d["hd_last_token"] + d["hd_last_mean"])
                labels.append(dt["label"])
                pre.append(score)
                if dt["label"] == 1:
                    psg_bi = 1
                    psg_not_bi += 1
            psg_not_bi /= len(dts)
            psgscore = mlp.eval(hd[k]["passage"]["hd_last_token"] + hd[k]["passage"]["hd_last_mean"])    
            psglabels.append(psg_bi)
            psgpre.append(psgscore)
            psglabelsbysent.append(psg_not_bi)
        roc_auc_hallu_s = get_AUC(pre, labels)
        roc_auc_fact_s = get_AUC([1-x for x in pre], [1-x for x in labels])
        roc_auc_hallu_p = get_AUC(psgpre, psglabels)
        roc_auc_fact_p = get_AUC([1-x for x in psgpre], [1-x for x in psglabels])
        corr = np.corrcoef(psgpre, psglabelsbysent)
        result_sent_halu["Our_score"][mo.split(".json")[0]] = roc_auc_hallu_s
        result_psg_corr["Our_score"][mo.split(".json")[0]] = corr[0][1]
        result_psg_halu["Our_score"][mo.split(".json")[0]] = roc_auc_hallu_p
        result_sent_corr["Our_score"][mo.split(".json")[0]] = np.corrcoef(pre, labels)[0][1]

import pandas as pd
df = pd.DataFrame(result_sent_halu)
df.to_excel(root_path + f"/result_sent_halu.xlsx")
df = pd.DataFrame(result_psg_corr)
df.to_excel(root_path + f"/result_psg_corr.xlsx")
df = pd.DataFrame(result_psg_halu)
df.to_excel(root_path + f"/result_psg_halu.xlsx")
df = pd.DataFrame(result_sent_corr)
df.to_excel(root_path + f"/result_sent_corr.xlsx")
        


