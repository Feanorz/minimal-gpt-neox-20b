import copy
import gc
import os
from tqdm import auto as tqdm_lib
import time
from torch.multiprocessing import Pool

import torch
import tokenizers

import minimal20b.model as model20b
from minimal20b.constants import Args20b, ArgsDummy

# Dynamic precision only works if half precision is also enabled
if Args20b.dynamic_precision:
    assert Args20b.half_precision

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

    if Args20b.half_precision or Args20b.dynamic_precision:
        model = model.half().to_empty(device=device)
    else:
        model = model.float().to_empty(device=device)

    pbar.update(1)

    # Move first n layers to GPU.
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")

    if Args20b.gpu_layers:
        for i, layer in enumerate(model.layer_list):
            if i < Args20b.gpu_layers:
                layer.to(gpu).half()
            else:
                layer.to(cpu).float()


    second_layer_list = []
    # Load transformer layers
    for layer_i in range(Args20b.num_layers):
        pbar.set_description(f"Loading layer {layer_i}")
        st = time.time()
        state_dict = load_layer(checkpoint_path, layer_i)
        st2 = time.time()
        if Args20b.dynamic_precision:
            second_layer_list.append(copy.deepcopy(state_dict))  # Already float16
        else:
            model.layer_list[layer_i].load_state_dict(state_dict)
        # torch.cuda.synchronize(device=torch.device("cuda:0"))
        end = time.time()

        print()
        print("Time to load file:", st2 - st)
        print("Time to load state:", end - st2)
        #del state_dict
        pbar.update(1)


    if Args20b.dynamic_precision:
        pbar.set_description(f"Casting model to float")
        model.second_layer_list = second_layer_list
        # Cast to float without assigning large amounts of memory
        for layer in model.layer_list:
            layer.float().to_empty(device=torch.device("cpu"))


    # Input and output embeddings, always have to be float
    if Args20b.half_precision:
        model.embed_in.float()
        model.final_layer_norm.float()
        model.logits_out.float()

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

    # for layer in model.layer_list:
    #     st = time.time()
    #
    #     #layer.to_empty(device=torch.device("cpu"))
    #     layer.load_state_dict(empty_states)
    #
    #     end = time.time() - st
    #     print("Time to convert layer to empty:", end)
    #
    # assert 1 == 2


    # if Args20b.dynamic_precision:
    #     pbar.set_description(f"Setting up duplicate layers")
    #     model.half()
    #
    #     # Duplicate Layer
    #     second_state_dict = []
    #     for layer in model.layer_list:
    #         second_state_dict.append(copy.deepcopy(layer.state_dict()))
    #         layer.to_empty(device=torch.device("cpu"))
    #
    #     model.float().to_empty(device=torch.device("cpu"))
    #     model.second_layer_list = second_state_dict

        # Empty Layers
        # layer = next(iter(model.layer_list))
        # state_dict = layer.state_dict()
        # for name, state in state_dict.items():
        #     state_dict[name] = torch.empty_like(state)

        #empty_states = copy.deepcopy(state_dict)
        # model.empty_layer = state_dict




    gc.collect()

    return model



def float_fun(x):
    x.float()
    return None


def load_layer(checkpoint_path, layer_i):

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
