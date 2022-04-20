import os
from tqdm import auto as tqdm_lib
import time

import torch
import tokenizers

import minimal20b.model as model20b
from minimal20b.constants import Args20b, ArgsDummy

DTYPE = torch.float32
def create_model(checkpoint_path, use_cache=False, device=torch.device("cpu")):
    """
    To prevent allocation memory on CPU, we initialize on 'meta' and individually
    port each module over to 'device' as we load each state dict.

    :param checkpoint_path: Path to the checkpoint folder
    :param use_cache: whether to use cache (i.e. for efficient generation)
    :param device: device that you want the model to end up on
    :return: model
    """
    # Instantiate model
    pbar = tqdm_lib.tqdm(total=48)
    pbar.set_description("Instantiating model (~1 min)")
    model = model20b.NeoX20BModel(Args20b, use_cache=use_cache, device="cpu")
    model = model.to_empty(device=device)
    pbar.update(1)

    # Load transformer layers
    for layer_i in range(Args20b.num_layers):
        pbar.set_description(f"Loading layer {layer_i}")
        st = time.time()
        state_dict = load_layer(checkpoint_path, layer_i, dtype=DTYPE)
        st2 = time.time()
        model.layer_list[layer_i].load_state_dict(state_dict)
        end = time.time()
        print()
        print("Time to load file:", st2 - st)
        print("Time to load state:", end - st2)
        pbar.update(1)

    # Load input embedding
    pbar.set_description(f"Loading input embedding")
    in_embedding = torch.load(os.path.join(checkpoint_path, "0_model_states.pt"))
    model.embed_in.load_state_dict(in_embedding)
    del in_embedding
    pbar.update(1)

    # Load final layer norm
    pbar.set_description(f"Loading final layer norm")
    final_layer_norm = torch.load(os.path.join(checkpoint_path, "47_model_states.pt"))
    model.final_layer_norm.load_state_dict(final_layer_norm)
    del final_layer_norm
    pbar.update(1)

    # Load output embedding
    pbar.set_description(f"Loading output embedding")
    logits_out = torch.load(os.path.join(checkpoint_path, "48_model_states.pt"))
    model.logits_out.load_state_dict(logits_out)
    del logits_out
    pbar.update(1)
    pbar.set_description("Done.")

    return model


def load_layer(checkpoint_path, layer_i, dtype=torch.float32):

    filename = f"{layer_i}.pt"
    loaded = torch.load(os.path.join(checkpoint_path, filename))

    # # Convert to fp32
    # for name, value in loaded.items():
    #     loaded[name] = value.type(dtype)

    return loaded




def create_dummy_model(use_cache=False, device=torch.device("cpu")):
    model = model20b.NeoX20BModel(ArgsDummy, use_cache=use_cache).half().to(device)
    return model


def create_tokenizer(tokenizer_path):
    return tokenizers.Tokenizer.from_file(tokenizer_path)
