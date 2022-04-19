import torch.nn as nn
import torch
import time

@torch.jit.script
def gelu(x):
    val_16 = 0.79788456 * x * (1 + 0.044715 * x * x)
    val_32 = val_16.type(torch.float32)
    return x * 0.5 * (1.0 + torch.tanh(val_32).type(torch.float16))



class MLP(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        ff_dim = 4 * args
        self.dense_h_to_4h = nn.Linear(args, ff_dim, device=device)
        self.dense_4h_to_h = nn.Linear(ff_dim, args, device=device)
        self.act = torch.nn.GELU()

    def forward(self, hidden_states):
        weight = self.dense_4h_to_h.weight.tolist()
        with open("./weights", "w") as f:
            f.write(str(weight))
        print("Written")
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.act(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


pony = MLP(6144)
x = torch.rand(6144*24).view(-1, 6144)
start = time.time()
out = pony(x)
print("Time taken:", time.time() - start)
print(out.shape)





