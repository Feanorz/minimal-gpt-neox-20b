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

for param in testclass.state_dict():
    print(param)

saved = copy.deepcopy(testclass.state_dict())
testclass.param1.weight = None

print("New parameters")
for param in testclass.state_dict():
    print(param)

x = testclass.load_state_dict(saved)
print(x)
for param in testclass.state_dict():
    print(param)
