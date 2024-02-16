import torch
import torch.nn.functional as F

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]

def chat_change(dialog,
        tokenizer):
    prompt_tokens = []
    unsafe_requests = []
    
    unsafe_requests.append(
        any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
    )
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    # print(dialog)
    dialog_tokens = sum(
        [
            tokenizer.encode(
                f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
            ) + [2]
            for prompt, answer in zip(
                dialog[::2],
                dialog[1::2],
            )
        ],
        [],
    )
    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"
    dialog_tokens += tokenizer.encode(
        f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
    )
    prompt_tokens.append(dialog_tokens)
    return prompt_tokens

def chat_change_with_answer(dialog, answer_,
        tokenizer):
    prompt_tokens = []
    unsafe_requests = []
    unsafe_requests.append(
        any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
    )
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    # print(dialog)
    dialog_tokens = sum(
        [
            tokenizer.encode(
                f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
            ) + [2]
            for prompt, answer in zip(
                dialog[::2],
                dialog[1::2],
            )
        ],
        [],
    )
    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"
    dialog_tokens += tokenizer.encode(
        f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST} {answer_.strip()}",
    )
    prompt_tokens.append(dialog_tokens)
    return prompt_tokens

def generate_output(model_family, model, tokenizer, config, text, answer=None):
    if "chat" not in model_family:
        assert answer is None
        input_id = tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()
    else:
        if answer is None:
            input_id = chat_change([{"role": "user", "content": text.strip()}], tokenizer)
        else:
            input_id = chat_change_with_answer([{"role": "user", "content": text.strip()}], answer.strip(), tokenizer)
    output = model.generate(torch.tensor(input_id).to(model.device), **config)
    return output


def get_pe(logit, id_, start_at):
    probabilities = F.softmax(logit, dim=2)
    log_probabilities = torch.log(probabilities)
    entropy = -probabilities * log_probabilities
    entropy_sum = torch.sum(entropy, dim=-1)

    pl = []
    el = []
    for i, idx in enumerate(id_[1:]):
        if i < start_at - 1:
            continue
        pl.append(probabilities[0][i][idx].item())
        el.append(entropy_sum[0][i].item())
    return pl, el