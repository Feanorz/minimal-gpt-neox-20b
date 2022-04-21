import torch.nn as nn
import torch
import time

class Test(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.param1 = torch.nn.parameter.Parameter(torch.tensor([1., 2., 3.], dtype=torch.float32))
        self.tensor1 = torch.tensor([1., 2., 3.], dtype=torch.float32)

    def forward(self, x):
        return x + self.param1


test_class = Test(1)

states = test_class.state_dict()
print(states, test_class.param1)

states["param1"] = torch.tensor([2., 4., 6.], dtype=torch.float16)
print(states)

print()
test_class.param1 = None
states2 = test_class.state_dict()
print(states2, test_class.param1)

test_class.load_state_dict(states)

states = test_class.state_dict()
print(states, test_class.param1)
