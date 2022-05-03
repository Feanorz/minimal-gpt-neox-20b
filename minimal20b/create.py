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
# If full_gpu is enabled, gpu must be used, half precision must be enabled and dynamic precision must be disabled
# if Args20b.full_gpu:
#     assert Args20b.half_precision
#     assert not Args20b.dynamic_precision
#     assert Args20b.gpu_layers
# If using GPU without full_gpu, half precision and dynamic precision must be off
if Args20b.gpu_layers and not Args20b.full_gpu:
    assert not Args20b.half_precision
    assert not Args20b.dynamic_precision
if Args20b.use_gpu:
    assert Args20b.gpu_layers != 0


def create_model(checkpoint_path, use_cache=False):
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

    # Move first n layers to GPU.
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")


    if Args20b.use_gpu:
        for i, layer in enumerate(model.layer_list):
            if i < Args20b.gpu_layers:
                layer.half().to_empty(device=gpu)#.half()
            elif Args20b.full_gpu:
                layer.half().to_empty(device=cpu)
            else:
                layer.float().to_empty(device=cpu)
    # CPU only
    elif Args20b.half_precision:
        model = model.half().to_empty(device=cpu)
    elif Args20b.full_precision_cpu:
        # Standard fp32 CPU implementation
        model = model.float().to_empty(device=cpu)
    else:
        assert 1 == 2

    pbar.update(1)


    second_layer_list = []
    # Load transformer layers
    for layer_i in range(Args20b.num_layers):
        pbar.set_description(f"Loading layer {layer_i}")

        state_dict = load_layer(checkpoint_path, layer_i)
        if Args20b.dynamic_precision:
            second_layer_list.append(copy.deepcopy(state_dict))  # Already float16
        elif Args20b.full_gpu and layer_i >= Args20b.gpu_layers:
            second_layer_list.append(copy.deepcopy(state_dict))
        else:
            model.layer_list[layer_i].load_state_dict(state_dict)
        del state_dict
        pbar.update(1)


    if Args20b.full_gpu:
        for layer in second_layer_list:
            if layer is not None:
                for param in layer.values():
                    param.pin_memory()

    model.second_layer_list = [None for _ in range(Args20b.gpu_layers)] + second_layer_list

    if Args20b.dynamic_precision:
        pbar.set_description(f"Casting model to float for dynamic precision")
        #model.second_layer_list = second_layer_list
        # Cast to float without assigning large amounts of memory
        for layer in model.layer_list:
            layer.float().to_empty(device=cpu)

        if Args20b.use_state_dict:
            for layer in model.layer_list:
                state_dict = layer.state_dict()
                for name, state in state_dict.items():
                    state_dict[name] = torch.empty_like(state)
                model.empty_layer = state_dict
                break


    # Input and output embeddings, always have to be float
    if Args20b.half_precision:
        pbar.set_description(f"Casting IO layers to float")
        model.embed_in.to(device=torch.device(cpu), dtype=torch.float32)
        model.final_layer_norm.to(device=torch.device(cpu), dtype=torch.float32)
        model.logits_out.to(device=torch.device(cpu), dtype=torch.float32)

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

    #model.float()

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
