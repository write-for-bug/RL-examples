"""
torch.gather 函数根据提供的索引张量，从输入张量中沿指定维度收集元素，生成一个新的张量。
2维tensor的gather示例
"""
import torch

# 输入张量
input_tensor = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# 索引张量，指定要提取的元素位置
index_tensor = torch.tensor([
    [0, 2 ,1],#第一行的0和2号元素：1，3
    [1, 0, 2],#第二行的1和0号元素：5，4
    [2, 1, 0] #第三行的2和1号元素：8，7
])

# 沿着第1维（列）收集元素
gathered1 = torch.gather(input_tensor, dim=1, index=index_tensor)#沿着列
print(gathered1)
# 输出:
# tensor([[1, 3],
#         [5, 4],
#         [9, 8]])
gathered2 = torch.gather(input_tensor, dim=0, index=index_tensor)#沿着行
print(gathered2)
print(input_tensor[:,0])