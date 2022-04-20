import minimal20b
import torch
print(torch.get_num_threads())
torch.set_num_threads(32)
print(torch.get_num_threads())

model = minimal20b.create_model(
    "/mnt/ssd/global_step150000",
    use_cache=True,
    device="cpu",
)
print("Model created")
print("________________________")

tokenizer = minimal20b.create_tokenizer(
    "/mnt/ssd/20B_tokenizer.json",
)

print("Doing Inference")
with torch.inference_mode():
    output = minimal20b.greedy_generate_text(
        model, tokenizer,
        "Did you ever hear the tragedy of Darth Plagius the Wise? I thought not. Its not a story the Jedi would tell you.",
        max_seq_len=500,
    )

    print(output)
