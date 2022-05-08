import torch
import torch.nn as nn
from tqdm import auto as tqdm_lib
import time
from minimal20b.constants import Args20b


def sample_generate(model: nn.Module, input_ids: torch.Tensor, max_seq_len: int, verbose=True):
    initial_input_length = input_ids.shape[1]
    current_input_ids = input_ids
    layer_past = None
    layer_past_length = 0
    all_token_ids = input_ids.tolist()
    batch_size = len(all_token_ids)

    trange = range(initial_input_length, max_seq_len)

    for _ in trange:
        print()
        st = time.time()

        input_length = current_input_ids.shape[1]
        model_out, layer_past = model(
            current_input_ids,
            layer_past=layer_past,
        )

        if Args20b.sample_output:
            output_distribution = torch.distributions.Categorical(logits=model_out[:, -1] / Args20b.sample_temp)
            prediction = output_distribution.sample()
        else:
            prediction = model_out[:, -1].argmax(-1)

        predicted_token_ids = prediction
        current_input_ids = predicted_token_ids[:, None]
        for i in range(batch_size):
            all_token_ids[i].append(predicted_token_ids[i])
        layer_past_length += input_length

        print()
        print("                                             Generation complete, time taken:", time.time() - st)
        print()
        yield all_token_ids
    # return all_token_ids


# Sample output of model. Method of sampling given in constants.py
def sample_generate_text(model: nn.Module,
                         tokenizer,
                         initial_str: str,
                         max_seq_len: int,
                         device=torch.device("cpu"),
                         verbose=True):
    print("Generating Text")
    print()
    tokenized = tokenizer.encode(initial_str)
    input_ids = torch.LongTensor([tokenized.ids, tokenized.ids]).to(device)
    all_token_ids = sample_generate(model=model, input_ids=input_ids, max_seq_len=max_seq_len, verbose=verbose)

    last = None
    while True:
        prediction = next(all_token_ids, None)
        if prediction is None:
            break
        last = prediction
        print()
        print(tokenizer.decode(prediction[0]))

    return tokenizer.decode(last[0])
