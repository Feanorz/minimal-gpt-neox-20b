import torch.nn as nn
import torch
device = torch.device("cuda:0")
import time


print("start")
print(1)
x = torch.ones(5000000, dtype=torch.float16, device=device)


time.sleep(10)
print(x)


