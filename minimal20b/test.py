import torch.nn as nn
import torch
import time
from torch.multiprocessing import Process, Queue
import copy

class TestClass(torch.nn.Module):

    def __init__(self):
        super(TestClass, self).__init__()
        self.param1 = nn.Linear(2, 10)
        self.param2 = torch.nn.parameter.Parameter(torch.tensor([1.]))


    def forward(self, x):
        with torch.no_grad():
            return self.param1(x)



testclass = TestClass()
testclass.half()
print(testclass.state_dict())
print()
for name, param in testclass.named_parameters():
    param.to(device=torch.device("cuda:0"))
    print(param.is_pinned())
    param.to(device=torch.device("cpu"))
    param.pin_memory()
    print(param.is_pinned())

    #torch.nn.init.ones_(param)
    param.data = torch.tensor([1.], dtype=torch.float32)
#
print()
print()
# for param in testclass.parameters():
#     print(param)
# param = getattr(testclass.param1, "weight")
# torch.nn.init.ones_(param)
#
# param = getattr(testclass.param1, "bias")
# torch.nn.init.ones_(param)
#print(param)

#setattr(testclass.param1, "weight", None)
#
# for name, param in testclass.named_parameters():
#     print(name, param)
#
# print()
# print()
#
# # out = testclass(torch.tensor([1., 2.]))
# # print(out)
