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
tokenizer = minimal20b.create_tokenizer(
    "/mnt/ssd/20B_tokenizer.json",
)

with torch.inference_mode():
    print("Doint Inference")
    output = minimal20b.greedy_generate_text(
        model, tokenizer,
        "GPTNeoX20B is a 20B-parameter autoregressive Transformer model developed by EleutherAI.",
        max_seq_len=27,
    )

    print(output)
