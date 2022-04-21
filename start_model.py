import minimal20b
import torch
print("Current threads:")
print("   ", torch.get_num_threads())
torch.set_num_threads(32)
print("New Threads:")
print("   ", torch.get_num_threads())

model = minimal20b.create_model(
    "/mnt/global_step150000/compressed",
    use_cache=True,
    device="cpu",
)
print("Model created")
print("________________________")

tokenizer = minimal20b.create_tokenizer(
    "/mnt/20B_tokenizer.json",
)

print("Doing Inference")
with torch.inference_mode():
    output = minimal20b.greedy_generate_text(
        model, tokenizer,
        "I am the Senate, said Palpatine.",
        max_seq_len=500,
    )

    print(output)
