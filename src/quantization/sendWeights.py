import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
import torchsummary
from dataset import get_data_loader_siena
from utils.parser_util import get_parser
from einops.layers.torch import Rearrange
import os
from few_shot_train import init_vit
from torch.utils.data import DataLoader
from few_shot.prototypical_loss import get_prototypes, prototypical_evaluation, prototypical_evaluation_per_patient
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
import time
from tqdm import tqdm
import numpy as np
from few_shot_train import get_support_set_per_patient, init_seed


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
                nn.MultiheadAttention(q_dim*n_heads, n_heads, dropout=0.2, bias=True, batch_first=True),
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
          x = self.quant(x.squeeze())
        # x = self.layer_norm_patch1(fake_input.squeeze())
        # x = self.embedding(x)
        # x = self.layer_norm_patch2(x)
        # x = self.dropout_patch(x)

        for modules in self.layers[:]:
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

        x = x[:, 0, :]
        x = self.to_latent(x)

        x = self.mlp_head(x)
        if self.q:
          x = self.dequant(x)

        return x


options = get_parser().parse_args()
init_seed(options)


def get_trained_model():
    exp_root = options.base_learner_root
    model_path = os.path.join('../', exp_root, 'best_model.pth')
    model = init_vit(options)
    model.load_state_dict(torch.load(model_path))
    return model


def send_weights(source_model, target_model):
    target_model.mlp_head[0].weight.data = source_model.mlp_head[0].weight
    target_model.mlp_head[0].bias.data = source_model.mlp_head[0].bias

    target_model.mlp_head[1].weight.data = source_model.mlp_head[1].weight
    target_model.mlp_head[1].bias.data = source_model.mlp_head[1].bias

    target_model.layer_norm_patch1.weight.data = source_model.to_patch_embedding[1].weight
    target_model.layer_norm_patch1.bias.data = source_model.to_patch_embedding[1].bias

    target_model.embedding.weight.data = source_model.to_patch_embedding[2].weight
    target_model.embedding.bias.data = source_model.to_patch_embedding[2].bias

    target_model.layer_norm_patch2.weight.data = source_model.to_patch_embedding[3].weight
    for l in range(4):
        target_model.layers[l][0].weight.data =  source_model.transformer.layers[l][0].norm.weight
        target_model.layers[l][0].bias.data =  source_model.transformer.layers[l][0].norm.bias
        target_model.layers[l][1].out_proj.weight.data = source_model.transformer.layers[l][0].fn.to_out[0].weight
        target_model.layers[l][1].out_proj.bias.data = source_model.transformer.layers[l][0].fn.to_out[0].bias
        target_model.layers[l][3].weight.data = source_model.transformer.layers[l][1].norm.weight
        target_model.layers[l][3].bias.data = source_model.transformer.layers[l][1].norm.bias
        target_model.layers[l][4].weight.data = source_model.transformer.layers[l][1].fn.net[0].weight
        target_model.layers[l][4].bias.data = source_model.transformer.layers[l][1].fn.net[0].bias
        target_model.layers[l][7].weight.data = source_model.transformer.layers[l][1].fn.net[3].weight
        target_model.layers[l][7].bias.data = source_model.transformer.layers[l][1].fn.net[3].bias

        target_model.layers[l][1].in_proj_weight.data = source_model.transformer.layers[l][0].fn.to_qkv.weight
        target_model.layers[l][1].in_proj_bias.data = torch.zeros(48)
    #
    # for param1_key in dict_params_source.keys():
    #     print(param1_key, dict_params_source[param1_key].shape, dict_params_source[param1_key].mean())
    #
    # print('*'*50)
    # for param2_key in dict_params_target.keys():
    #     print(param2_key, dict_params_target[param2_key].shape, dict_params_target[param2_key].mean())
    return


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def main():
    model = get_trained_model()
    net = ViTInferenceNet(q=True)
    print(torchsummary.summary(model.cpu(), (1, 3200, 15), batch_size=-1, device='cpu'))
    model.eval()
    net.eval()
    send_weights(model, net)
    hook = model.dropout.register_forward_hook(get_activation('transformer_input'))
    source_output = model(torch.rand((32, 1, 3200, 15)))
    target_output = net(activation['transformer_input'])
    print("Error", torch.sum(torch.abs(target_output - source_output)))
    hook.remove()

    all_patients = [0, 1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17]
    test_patient_ids = [p for p in all_patients if p not in options.excluded_patients]
    test_dataloader = get_data_loader_siena(batch_size=32, patient_ids=test_patient_ids, save_dir=options.siena_data_dir)
    train_dataloader = get_data_loader_siena(batch_size=32, patient_ids=options.patients, save_dir=options.siena_data_dir)
    test(options, test_dataloader=test_dataloader, model=model, print_results=True, target_model=net)

    net.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    torch.ao.quantization.prepare(net, inplace=True)
    print('Post Training Quantization Prepare: Inserting Observers')

    test(options, test_dataloader=test_dataloader, model=model, print_results=True, target_model=net)
    print('Post Training Quantization: Calibration done')
    torch.ao.quantization.convert(net, inplace=True)
    print('Post Training Quantization: Convert done')
    print('\n Conv1: After fusion and quantization \n\n', net.layers[3][1])
    test(options, test_dataloader=test_dataloader, model=model, print_results=True, target_model=net)


def test(opt, test_dataloader, model, print_results=False, target_model = None):
    """
    Test the model trained with the prototypical learning algorithm
    """
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    start_time = time.time()

    model.eval()

    x_support_set_all, y_support_set_all = get_support_set_per_patient(opt.num_support_val,
                                                               data_dir=opt.siena_data_dir,
                                                               patient_ids=opt.patients)
    prototypes_all = []

    hook = model.dropout.register_forward_hook(get_activation('transformer_input'))
    for x_support_set, y_support_set in zip(x_support_set_all, y_support_set_all):
        x_support_set = torch.tensor(x_support_set).to(device)
        y_support_set = torch.tensor(y_support_set).to(device)

        x = x_support_set.reshape((x_support_set.shape[0], 1, -1, x_support_set.shape[3]))
        model_output = model(x)
        if target_model is not None:
            transformer_input = activation['transformer_input']
            model_output = target_model(transformer_input)
        prototypes = get_prototypes(model_output, target=y_support_set)
        print("Prototypes", prototypes)
        prototypes_all.append(prototypes)

    hook.remove()
    predict = []
    predict_prob = []
    true_label = []
    hook = model.dropout.register_forward_hook(get_activation('transformer_input'))
    for i, batch in tqdm(enumerate(test_dataloader)):
        x, y = batch
        x, y = x.to(device), y.to(device)

        x = x.reshape((x.shape[0], 1, -1, x.shape[3]))

        model_output = model(x)
        # print("Source model output", model_output.shape, model_output)
        if target_model is not None:
            # print("Activation", activation['transformer_input'])
            transformer_input = activation['transformer_input']
            model_output = target_model(transformer_input)
            # print("Target model output", model_output.shape, model_output)

        prob, output = prototypical_evaluation_per_patient(prototypes_all, model_output)
        predict.append(output.detach().cpu().numpy())
        predict_prob.append(prob.detach().cpu().numpy())
        true_label.append(y.detach().cpu().numpy())
    hook.remove()
    predict = np.hstack(predict)
    predict_prob = np.hstack(predict_prob)
    true_label = np.hstack(true_label)

    # Placeholder for results
    results = {
        "seed": opt.manual_seed,
        "num_support": opt.num_support_val,
        "skip_base_learner": opt.skip_base_learner,
        "skip_finetune": opt.skip_finetune,
        "patients": opt.patients,
        "finetune_patients": opt.finetune_patients,
        "excluded_patients": opt.excluded_patients,
        "auc": roc_auc_score(true_label, predict_prob)
    }

    if print_results:
        print(results)


if __name__ == '__main__':
    main()



# x, y = next(iter(test_dataloader))
# x = x.reshape((x.shape[0], 1, -1, x.shape[3]))
# print("X reshape", x.shape)
# x_rearrange = rearrange(x)
# print(x_rearrange.shape)
# model_output = net(x_rearrange)

