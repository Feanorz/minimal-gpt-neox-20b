import torch
import os

checkpoint_path ="/mnt/20B_checkpoints/global_step150000"

for layer_i in range(44):
        filename_tp1 = f"layer_{layer_i + 2:02d}-model_00-model_states.pt"
        filename_tp2 = f"layer_{layer_i + 2:02d}-model_01-model_states.pt"
        loaded_tp1 = torch.load(os.path.join(checkpoint_path, filename_tp1))
        loaded_tp2 = torch.load(os.path.join(checkpoint_path, filename_tp2))
        state_dict = {}
        # Good
        # Keys where we concatenate on the second dim
        for key in [
            "attention.dense.weight",
            "mlp.dense_4h_to_h.weight",
        ]:
            state_dict[key] = torch.cat([loaded_tp1[key], loaded_tp2[key]], dim=1)
        # Mapping individual split weights to custom split implementations
        # Layer Norms
        # Choose 1
        state_dict["input_layernorm.weight"] = (
            loaded_tp1["input_layernorm.weight"] + loaded_tp2["input_layernorm.weight"]) / 2
        state_dict["input_layernorm.bias"] = (
            loaded_tp1["input_layernorm.bias"] + loaded_tp2["input_layernorm.bias"]) / 2
        state_dict["post_attention_layernorm.weight"] = (
            loaded_tp1["post_attention_layernorm.weight"] + loaded_tp2["post_attention_layernorm.weight"]) / 2
        state_dict["post_attention_layernorm.bias"] = (
            loaded_tp1["post_attention_layernorm.bias"] + loaded_tp2["post_attention_layernorm.bias"]) / 2
        # LinearWithTPMerge
        state_dict["mlp.dense_h_to_4h.weight"] = torch.cat([
            loaded_tp1["mlp.dense_h_to_4h.weight"],
            loaded_tp2["mlp.dense_h_to_4h.weight"],
        ], dim=0)
        state_dict["mlp.dense_h_to_4h.bias"] = torch.cat([
            loaded_tp1["mlp.dense_h_to_4h.bias"],
            loaded_tp2["mlp.dense_h_to_4h.bias"],
        ], dim=0)
        state_dict["attention.query_key_value.weight"] = torch.cat([
            loaded_tp1["attention.query_key_value.weight"],
            loaded_tp2["attention.query_key_value.weight"],
        ], dim=0)
        state_dict["attention.query_key_value.bias"] = torch.cat([
            loaded_tp1["attention.query_key_value.bias"],
            loaded_tp2["attention.query_key_value.bias"],
        ], dim=0)
        # LinearWithTPSplitBias
        state_dict["mlp.dense_4h_to_h.bias"] = (
            loaded_tp1["mlp.dense_4h_to_h.bias"]
            + loaded_tp2["mlp.dense_4h_to_h.bias"]
        )
        state_dict["attention.dense.bias"] = (
            loaded_tp1["attention.dense.bias"]
            + loaded_tp2["attention.dense.bias"]
        )
        # Just take one
        state_dict["attention.rotary_emb.inv_freq"] = loaded_tp1["attention.rotary_emb.inv_freq"]

        torch.save(state_dict, f'{checkpoint_path}/compressed/{layer_i}.pt')

        print(f'Layer {layer_i} Completed')


loaded_tp1 = torch.load(os.path.join(checkpoint_path, "layer_00-model_00-model_states.pt"))
loaded_tp2 = torch.load(os.path.join(checkpoint_path, "layer_00-model_01-model_states.pt"))
out = {"weight": torch.cat([
    loaded_tp1["word_embeddings.weight"],
    loaded_tp2["word_embeddings.weight"],
], dim=0)}
torch.save(out, f'{checkpoint_path}/compressed/0_model_states.pt')


# Load final layer norm
loaded_tp1 = torch.load(os.path.join(checkpoint_path, "layer_47-model_00-model_states.pt"))
loaded_tp2 = torch.load(os.path.join(checkpoint_path, "layer_47-model_01-model_states.pt"))
out = {
    "weight": (loaded_tp1["norm.weight"] + loaded_tp2["norm.weight"])/2,
    "bias": (loaded_tp1["norm.bias"] + loaded_tp2["norm.bias"])/2,
}
torch.save(out, f'{checkpoint_path}/compressed/47_model_states.pt')



# Load output embedding
loaded_tp1 = torch.load(os.path.join(checkpoint_path, "layer_48-model_00-model_states.pt"))
loaded_tp2 = torch.load(os.path.join(checkpoint_path, "layer_48-model_01-model_states.pt"))
out = {
    "weight": torch.cat([
        loaded_tp1["final_linear.weight"],
        loaded_tp2["final_linear.weight"],
    ], dim=0),
}
torch.save(out, f'{checkpoint_path}/compressed/48_model_states.pt')


print("All DOne")