import torch.nn as nn
import torch
import time

# class Test(nn.Module):
#     def __init__(self, i):
#         super().__init__()
#         self.param1 = torch.nn.parameter.Parameter(torch.tensor([1., 2., 3.], dtype=torch.float32))
#         self.tensor1 = torch.tensor([1., 2., 3.], dtype=torch.float32)
#
#     def forward(self, x):
#         return x + self.param1
#
#
# test_class = Test(1)
#
# states = test_class.state_dict()
# print(states, list(test_class.parameters()))
# def chunk(x, n):
#     length = len(x)

# Multiprocessing init of transformer layers
self.layer_list = nn.ModuleList([])
with Pool(4) as p:
    layer_inputs = [(args, use_cache) for _ in range(args.num_layers)]

    for i in range(0, args.num_layers, 4):
        batch = layer_inputs[i:i + 4]
        transformer_layers = p.starmap(TransformerLayer, layer_inputs)

        for transformer_layer in transformer_layers:
            self.layer_list.append(transformer_layer)
print("All layers initialised")



