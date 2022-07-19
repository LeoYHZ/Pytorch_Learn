import torch
import torch.nn.functional as F


input_data = torch.tensor([[1,2,0,3,1],
                           [0,1,2,3,1],
                           [1,2,1,0,0],
                           [5,2,3,1,1],
                           [2,1,0,1,1]])

kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

# 调整数据尺寸使其满足卷积的输入
input_data = torch.reshape(input_data, (1,1,5,5))
kernel = torch.reshape(kernel, (1,1,3,3))

print(input_data.shape)
print(kernel.shape)

# stride=1 时输出3*3的数据
output_data = F.conv2d(input_data, kernel, stride=1)
print(output_data)

# stride=2 时输出2*2的数据
output_data2 = F.conv2d(input_data, kernel, stride=2)
print(output_data2)

# stride=1 时加上padding来输出5*5的数据
output_data_p = F.conv2d(input_data, kernel, stride=1, padding=1)
print(output_data_p)