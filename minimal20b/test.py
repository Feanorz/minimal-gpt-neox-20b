import torch.nn as nn
import torch
import time
from torch.multiprocessing import Process, Queue
import copy

class TestClass(torch.nn.Module):

    def __init__(self):
        super(TestClass, self).__init__()
        self.param1 = nn.Linear(100, 100)


    def forward(self, x):
        return self.param1(x)


testclass = TestClass()

for param in testclass.parameters():
    print(param)

# saved = copy.deepcopy(testclass.state_dict())
#
# state_dict = testclass.state_dict()
# #state_dict["param1.weight"] = torch.tensor([1])
# print(testclass._modules)
# setattr(testclass, "_modules", state_dict)
# print(testclass._modules)
# print()
# print()
# print(testclass.state_dict())
# print("New parameters")
# for param, v in testclass.state_dict().items():
#     print(param, v)
#
# x = testclass.load_state_dict(saved)
# print(x)
# for param in testclass.state_dict():
#     print(param)
