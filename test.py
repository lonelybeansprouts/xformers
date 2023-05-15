from xformers.ops import memory_efficient_attention
import torch


q = torch.ones((1, 64, 1, 8)).half().cuda()
k = torch.ones((1, 64, 1, 8)).half().cuda()
v = torch.ones((1, 64, 1, 8)).half().cuda()


for i in range(64):
    for j in range(8):
        q[0, i, 0, j] = i * 10 + j

print(q)

y = memory_efficient_attention(q, k, v)

print(y)