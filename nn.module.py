import torch
import torch.nn as nn
import torch.nn.functional as F

class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1

# 创建所需要的Module模块
module_test = Module()

input_data = torch.tensor(1)
output_data = module_test(input_data)
print(output_data)

