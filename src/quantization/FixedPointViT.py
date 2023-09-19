import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

CLIP_VAL = 8
FRACTION_BITS = 12


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def make_fxp(source_weight):
    eps = 1 / (2**FRACTION_BITS)
    target_weight = torch.where(source_weight >= CLIP_VAL - eps, CLIP_VAL - eps, source_weight)
    target_weight = torch.where(target_weight < -(CLIP_VAL - eps),  -(CLIP_VAL - eps), target_weight)
    target_weight *= (2**FRACTION_BITS)
    target_weight = target_weight.to(torch.int)
    return target_weight.to(torch.float) / (2**FRACTION_BITS)


def save_signals(x, name, first_signal):
    with open("../../output/signal.cpp", "w" if first_signal else "a") as f:
        param_to_write = x.detach().cpu().numpy().reshape((-1,))
        param_to_write = param_to_write * (2 ** FRACTION_BITS)
        param_to_write = param_to_write.astype(np.int16)
        f.write("int16_t {}[{}] =".format(name, param_to_write.shape[0]))
        f.write("{")
        for elem in param_to_write:
            f.write(str(elem))
            f.write(", ")
        f.write("};\n")


def debug_fxp(x):
    param_to_write = x.detach().cpu().numpy().reshape((-1,))
    param_to_write = param_to_write * (2 ** FRACTION_BITS)
    param_to_write = param_to_write.astype(np.int32)
    print(param_to_write)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        norm = self.norm(x)
        norm = make_fxp(norm)
        return self.fn(norm, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()

        self.ff1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.ff2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ff1(x)
        x = make_fxp(x)
        x = self.gelu(x)
        x = make_fxp(x)
        x = self.drop1(x)
        x = make_fxp(x)
        x = self.ff2(x)
        x = make_fxp(x)
        x = self.drop2(x)
        x = make_fxp(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., layer_num = -1):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.projection = nn.Linear(inner_dim, dim)
        self.drop_projection = nn.Dropout(dropout)


    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # print(torch.sum(torch.abs(q - qkv[0])))
        q = make_fxp(q)
        k = make_fxp(k)
        v = make_fxp(v)
        # save_signals(q[0][0], "transformer_layers_{}_0_fn_q".format(self.layer_num), first_signal=False)
        # save_signals(k[0][0], "transformer_layers_{}_0_fn_k".format(self.layer_num), first_signal=False)
        # save_signals(k.transpose(-1, -2)[0][0], "transformer_layers_{}_0_fn_kT".format(self.layer_num), first_signal=False)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = make_fxp(dots)
        print("Shape ", dots.shape)

        attn = self.attend(dots)
        attn = make_fxp(attn)
        attn = self.dropout(attn)
        attn = make_fxp(attn)

        out = torch.matmul(attn, v)
        out = make_fxp(out)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        out = make_fxp(out)
        out = self.drop_projection(out)
        out = make_fxp(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for l in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, layer_num=l)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = make_fxp(x)
            # save_signals(x[0], "transformer_layers_{}_0_fn_add".format(i), first_signal=False)
            x = ff(x) + x
            x = make_fxp(x)
        return x

class FixedPointViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding_rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.to_patch_embedding_layer_norm1 = nn.LayerNorm(patch_dim)
        self.to_patch_embedding_linear = nn.Linear(patch_dim, dim)
        self.to_patch_embedding_layer_norm2 = nn.LayerNorm(dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head_layer_norm = nn.LayerNorm(dim)
        self.mlp_head_linear = nn.Linear(dim, num_classes)

    def forward(self, img, w, b):
        img = make_fxp(img)
        x = self.to_patch_embedding_rearrange(img)
        x = make_fxp(x)
        save_signals(x[0], "input_signal", first_signal=True)
        x = self.to_patch_embedding_layer_norm1(x)
        x = make_fxp(x)
        x = self.to_patch_embedding_linear(x)
        x = make_fxp(x)
        x = self.to_patch_embedding_layer_norm2(x)
        x = make_fxp(x)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = make_fxp(x)
        save_signals(x[0], "pos_embedding_signal", first_signal=False)
        x = self.dropout(x)
        x = make_fxp(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = make_fxp(x)
        x = self.mlp_head_layer_norm(x)
        x = make_fxp(x)
        x = self.mlp_head_linear(x)
        x = make_fxp(x)

        return x
