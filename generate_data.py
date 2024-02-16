import os

# --------------------------------------------- #
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
model_type = "7b"
model_family = "llamabase"

wiki_path = "./auto-labeled/wiki"
output_path = f"./auto-labeled/output/{model_family}{model_type}"

topk_first_token = 4
windows = 16
# --------------------------------------------- #


topk_next_token = topk_first_token

import torch
from utils.model import get_model
from utils.gen import chat_change_with_answer
from tqdm import tqdm
import json
import spacy

model, tokenizer, generation_config, at_id = get_model(model_type, model_family, 1)


if not os.path.exists(output_path):
    os.mkdir(output_path)


if "llama" in model_family or "baichuan" in model_family:
    st = "▁"
else:
    st = "Ġ"
    
    
nlp = spacy.load('en_core_web_sm')
prompt_chat = []

def delete_substrings(lst):
    substrings = []
    lst = list(set(lst))
    for s in lst:
        if any(s in o for o in lst if o != s):
            substrings.append(s)
    for s in substrings:
        lst.remove(s)
    return lst

def find_boundaries(text, words):
    boundaries = []
    for word in words:
        start = 0
        ntext = text
        while True:
            start = ntext.find(word)
            if start == -1:
                break
            end = start + len(word) - 1
            while start > 0 and ntext[start-1] != " ":
                start -= 1
            while end < len(ntext)-1 and ntext[end+1] != " ":
                end += 1
            boundaries.append("".join([ntext[i] for i in range(start, end+1)]))
            ntext = ntext[end+1:]
    return boundaries

def get_entities(text):
    entities_ = list(set([str(e) for e in nlp(text).ents]))
    entities_ = find_boundaries(text, entities_)
    entities = delete_substrings(entities_)
    all_entities = []
    for i in range(len(text)):
        for e in entities:
            if text[i:].startswith(e):
                all_entities.append((e, i))
                
    return all_entities


def find_first_and_next_token(text, e, idx, input_id, prompt=""):
    new_text = f"{text[:idx].strip()} {text[idx:].replace(e, e + ' @', 1).strip()}" 
    new_input_id = tokenizer(prompt + new_text.strip(), return_tensors='pt')['input_ids'].tolist()[0]
    for i in range(len(input_id[0])):
        if input_id[0][i] != new_input_id[i]:
            return []
    if model_family == "falcon":
        correct_id = tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()[0]
        ap = 0
        for i in range(len(new_input_id)):
            if i >= len(correct_id):
                return []
            if correct_id[i] != new_input_id[i]:
                next_token = correct_id[i]
                ap = i
                break
                
        first_token = new_input_id[len(input_id[0])]
        try:
            return [first_token, next_token, ap-1-len(input_id[0]), correct_id[ap:]]
        except:
            return []
    
    first_token = new_input_id[len(input_id[0])]
    if type(at_id) == list:
        at_position = len(new_input_id) - 1
        for i in range(len(new_input_id)):
            if new_input_id[i] < first_token:
                continue
            if new_input_id[i] in at_id:
                at_position = i
                break
    else:
        at_position = new_input_id.index(at_id)
    if at_position == len(new_input_id) - 1:
        return []
    next_token = new_input_id[at_position+1]
    return [first_token, next_token, at_position-len(input_id[0]), new_input_id[at_position+1:]]

def find_first_and_next_token_for_chat(text, e, idx, input_id):
    new_text = f"{text[:idx].strip()} {text[idx:].replace(e, e + ' @', 1).strip()}" 
    new_input_id = chat_change_with_answer(prompt_chat, new_text.strip(), tokenizer)[0]
    for i in range(len(input_id[0])):
        if input_id[0][i] != new_input_id[i]:
            return []
    first_token = new_input_id[len(input_id[0])]
    at_position = new_input_id.index(732)
    if at_position == len(new_input_id) - 1:
        return []
    next_token = new_input_id[at_position+1]
    return [first_token, next_token, at_position-len(input_id[0]), new_input_id[at_position+1:]]

def vicuna_prompt(title):
    return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Question: Tell me something about {title}.\nAnswer: \nASSISTANT: "

def chat_prompt(title):
    return [{"role": "user", "content": f"Question: Tell me something about {title}.\nAnswer: "}]

for data_type in ["train", "valid", "test"]:
    result = []

    with open(f"{wiki_path}/wiki_{data_type}.json", encoding='utf-8') as f:
        data = json.load(f)

    for ii, d in tqdm(enumerate(data)):

        text = " ".join(d["sentences"][:2])
        entities_ = []
        entities_ += get_entities(text)
        
        entities = []
        idx_ = []
        for e in entities_:
            if e[1] not in idx_:
                idx_.append(e[1])
                entities.append(e)
    
        mytexts = []
        new_entities = []
        original_entity = []
        ret = {
                "original_text": text,
                "title": d["title"]
            }
        prompt_chat = chat_prompt(d["title"])
        
        for e, idx in entities:
            if idx == 0 or e in d["title"]:
                continue
            
            if "chat" not in model_family:
                if model_family == "vicuna":
                    p_ = vicuna_prompt(d["title"])
                    input_id = tokenizer(p_ + text[:idx].strip(), return_tensors='pt')['input_ids'].tolist()
                else:
                    input_id = tokenizer(text[:idx].strip(), return_tensors='pt')['input_ids'].tolist()
                    p_ = ""
                tokens = find_first_and_next_token(text, e, idx, input_id, p_)
            else:
                input_id = chat_change_with_answer(prompt_chat, text[:idx].strip(), tokenizer)
                tokens = find_first_and_next_token_for_chat(text, e, idx, input_id)
            
            if not tokens:
                continue
            first_, next_, entity_len, last_id = tokens
        
            
            output = model.generate(torch.tensor(input_id).to(model.device), **generation_config)
            values, indices = torch.topk(output.scores[0], k=topk_first_token)
            if first_ in indices[0].tolist():
                continue
            sequences = output.sequences
            for i in range(entity_len+windows):
                output = model.generate(sequences, **generation_config)
                values, indices = torch.topk(output.scores[0], k=topk_next_token)
                if next_ in indices[0].tolist():
                    break
                sequences = output.sequences
            if next_ not in indices[0].tolist():
                continue
            new_sequence = sequences[0].tolist()
            new_entity_id = new_sequence[len(input_id[0]):]
            
            if model_family == "falcon":
                all_new_text_id = input_id[0] + [204, 43, 204] + new_entity_id + [204, 43, 204] + last_id
            elif type(at_id) == list:
                all_new_text_id = input_id[0] + [at_id[0]] + new_entity_id + [at_id[0]] + last_id
            else:
                all_new_text_id = input_id[0] + [at_id] + new_entity_id + [at_id] + last_id
            mytext = tokenizer.decode(all_new_text_id).replace("<s>", "").replace("</s>", "")
            new_entity = mytext[mytext.find("@")+1:mytext.rfind("@")].strip().lower()
            if any(ee.strip() in text.lower() for ee in new_entity.split(" ")) or e.lower() in new_entity:
                continue
            if model_family == "vicuna":
                mytext = mytext.split("ASSISTANT:")[-1].strip()
            if "chat" in model_family:
                mytext = mytext.split("[/INST]")[-1].strip()
            mytexts.append(mytext)
            new_entities.append(new_entity)
            original_entity.append((e, idx))
            
        ret["texts"] = mytexts
        ret["new_entities"] = new_entities
        ret["original_entities"] = original_entity
        result.append(ret)
    

    with open(f"{output_path}/data_{data_type}.json", "w+", encoding='utf-8') as f:
        json.dump(result, f, indent=4)