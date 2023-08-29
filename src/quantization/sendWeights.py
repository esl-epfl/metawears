import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
import torchsummary
from dataset import get_data_loader_siena
from utils.parser_util import get_parser
from einops.layers.torch import Rearrange
import os
from few_shot_train import init_vit


dim = 16
q_dim = 4
n_heads = 4
mlp_dim = 4
input_dim = 400


class ViTInferenceNet(nn.Module):
    def __init__(self, q = False):
        # By turning on Q we can turn on/off the quantization
        super(ViTInferenceNet, self).__init__()

        self.layers = nn.ModuleList([])
        for _ in range(4):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(q_dim*n_heads, n_heads, dropout=0.2, bias=True),
                nn.Dropout(0.2),

                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(mlp_dim, dim),
                nn.Dropout(0.2)
            ]))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 16)
        )
        self.layer_norm_patch1 = nn.LayerNorm(input_dim)
        self.layer_norm_patch2 = nn.LayerNorm(dim)
        self.dropout_patch = nn.Dropout(0.2)
        self.embedding = nn.Linear(input_dim, dim)
        self.to_latent = nn.Identity()
        self.ff = torch.nn.quantized.FloatFunctional()
        self.q = q
        if q:
          self.quant = QuantStub()
          self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.q:
          x = self.quant(x)
        x = self.layer_norm_patch1(x.squeeze())
        x = self.embedding(x)
        x = self.layer_norm_patch2(x)
        x = self.dropout_patch(x)
        for modules in self.layers:
            ln_out = modules[0](x)  # LayerNorm
            mha_out, _ = modules[1](ln_out, ln_out, ln_out)  # MultiHeadAttention
            projection = modules[2](mha_out)    # Projection + dropout
            x = self.ff.add(x, projection)

            ln_out = modules[3](x)  # LayerNorm
            ff1_out = modules[4](ln_out)  # Feed Forward 1
            gelu = modules[5](ff1_out)  # GELU
            dropout1 = modules[6](gelu)  # Dropout 1
            ff2_out = modules[7](dropout1)  # Feed Forward 2
            dropout2 = modules[8](ff2_out)  # Dropout 2
            x = self.ff.add(x, dropout2)

        x = x[:, 0]
        x = self.to_latent(x)

        x = self.mlp_head(x)
        if self.q:
          x = self.dequant(x)

        return x


rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 80, p2 = 5)

net = ViTInferenceNet(q=False)
print(torchsummary.summary(net, (1, 121, 400), batch_size=-1, device='cpu'))

options = get_parser().parse_args()


def get_trained_model():
    exp_root = options.base_learner_root
    model_path = os.path.join('../', exp_root, 'best_model.pth')
    model = init_vit(options)
    model.load_state_dict(torch.load(model_path))
    return model


def send_weights(source_model, target_model):
    mapping_layer_name = {
        "to_patch_embedding.1": "layer_norm_patch1",
        "to_patch_embedding.2": "embedding",
        "to_patch_embedding.3": "layer_norm_patch2",
        "transformer.layers.0.0.norm": "layers.0.0",
        "transformer.layers.0.0.fn.to_qkv.weight": "layers.0.0",

    }
    params1 = source_model.named_parameters()
    params2 = target_model.named_parameters()

    dict_params1 = dict(params1)
    dict_params2 = dict(params2)

    for param1_key in dict_params1.keys():
        print(param1_key, dict_params1[param1_key].shape)

    print('*'*50)
    for param2_key in dict_params2.keys():
        print(param2_key, dict_params2[param2_key].shape)


def main():
    model = get_trained_model()
    print(torchsummary.summary(model.cpu(), (1, 3200, 15), batch_size=-1, device='cpu'))
    send_weights(model, net)


if __name__ == '__main__':
    main()

test_dataloader = get_data_loader_siena(batch_size=32, patient_ids=[0], save_dir=options.siena_data_dir)


# x, y = next(iter(test_dataloader))
# x = x.reshape((x.shape[0], 1, -1, x.shape[3]))
# print("X reshape", x.shape)
# x_rearrange = rearrange(x)
# print(x_rearrange.shape)
# model_output = net(x_rearrange)

