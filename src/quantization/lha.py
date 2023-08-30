from vit_pytorch.vit import Attention
import torch.nn as nn
import torch
from einops import rearrange

attention = Attention(4, heads=2, dim_head=2, dropout=0)
mha = nn.MultiheadAttention(4, num_heads=2, dropout=0, bias=True, batch_first=True)

random_input = torch.rand((2,3,4))
# random_input_rearranged = rearrange(random_input, 'b n (h d) -> b h n d', h = 2)
# Q= torch.matmul(random_input, torch.ones((4,2)))
# K = Q
# V= Q
# print("K trans", torch.transpose(K, 1 , 2).shape)
# QKT = torch.matmul(Q, torch.transpose(K, 1 , 2))
# print("QKT", QKT)
# print("")

print("Attention", attention)
print("MHA", mha)
print("Attention", [(key, value) for key, value in attention.named_parameters()])
print("MHA", [(key, value) for key, value in mha.named_parameters()])

qkv_weight = torch.rand((12, 4))
out_weight = torch.rand((4, 4))
out_bias = torch.rand((4))
attention.to_qkv.weight.data = qkv_weight
attention.to_out[0].weight.data = out_weight
attention.to_out[0].bias.data =  out_bias

mha.in_proj_weight.data = qkv_weight
mha.in_proj_bias.data = torch.zeros((12))
mha.out_proj.weight.data = out_weight
mha.out_proj.bias.data = out_bias




activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


attention.attend.register_forward_hook(get_activation('qkv'))
print("Attention out ", attention(random_input))
print("MHA out", mha(random_input, random_input, random_input))
# print("Attention QKV", activation['qkv'])
