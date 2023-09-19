import numpy as np
import torch
import torch.nn as nn
import os
from few_shot_train import init_vit
from einops import rearrange, repeat
from FixedPointViT import FixedPointViT
from FixedPointViT import CLIP_VAL, FRACTION_BITS
import matplotlib.pyplot as plt
from few_shot.prototypical_loss import get_prototypes, prototypical_evaluation, prototypical_evaluation_per_patient
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
import time
from tqdm import tqdm
from few_shot_train import get_support_set_per_patient, init_seed
from dataset import get_data_loader_siena
from utils.parser_util import get_parser


options = get_parser().parse_args()
init_seed(options)
device = 'cuda:0' if torch.cuda.is_available() and options.cuda else 'cpu'

total_weights = torch.zeros(0).to(device)
total_activation = torch.zeros(0).to(device)


def make_fxp(source_weight):
    global total_weights

    eps = 1 / (2 ** FRACTION_BITS)
    target_weight = torch.where(source_weight >= CLIP_VAL - eps, CLIP_VAL - eps, source_weight)
    target_weight = torch.where(target_weight < -(CLIP_VAL - eps), -(CLIP_VAL - eps), target_weight)
    target_weight *= (2**FRACTION_BITS)
    target_weight = target_weight.to(torch.int)
    total_weights = torch.cat((total_weights, target_weight.reshape(-1, )))
    return target_weight.to(torch.float) / (2**FRACTION_BITS)


def send_weights(source_model, target_model):
    target_model.pos_embedding.data = make_fxp(source_model.pos_embedding)
    target_model.cls_token.data = make_fxp(source_model.cls_token)
    target_model.mlp_head_layer_norm.weight.data = make_fxp(source_model.mlp_head[0].weight)
    target_model.mlp_head_layer_norm.bias.data = make_fxp(source_model.mlp_head[0].bias)

    target_model.mlp_head_linear.weight.data = make_fxp(source_model.mlp_head[1].weight)
    target_model.mlp_head_linear.bias.data = make_fxp(source_model.mlp_head[1].bias)

    target_model.to_patch_embedding_layer_norm1.weight.data = make_fxp(source_model.to_patch_embedding[1].weight)
    target_model.to_patch_embedding_layer_norm1.bias.data = make_fxp(source_model.to_patch_embedding[1].bias)

    target_model.to_patch_embedding_linear.weight.data = make_fxp(source_model.to_patch_embedding[2].weight)
    target_model.to_patch_embedding_linear.bias.data = make_fxp(source_model.to_patch_embedding[2].bias)

    target_model.to_patch_embedding_layer_norm2.weight.data = make_fxp(source_model.to_patch_embedding[3].weight)
    target_model.to_patch_embedding_layer_norm2.bias.data = make_fxp(source_model.to_patch_embedding[3].bias)

    for l in range(4):
        target_model.transformer.layers[l][0].norm.weight.data = make_fxp(
            source_model.transformer.layers[l][0].norm.weight)
        target_model.transformer.layers[l][0].norm.bias.data = make_fxp(source_model.transformer.layers[l][0].norm.bias)
        target_model.transformer.layers[l][0].fn.projection.weight.data = make_fxp(
            source_model.transformer.layers[l][0].fn.to_out[0].weight)
        target_model.transformer.layers[l][0].fn.projection.bias.data = make_fxp(
            source_model.transformer.layers[l][0].fn.to_out[0].bias)
        target_model.transformer.layers[l][1].norm.weight.data = make_fxp(
            source_model.transformer.layers[l][1].norm.weight)
        target_model.transformer.layers[l][1].norm.bias.data = make_fxp(source_model.transformer.layers[l][1].norm.bias)
        target_model.transformer.layers[l][1].fn.ff1.weight.data = make_fxp(
            source_model.transformer.layers[l][1].fn.net[0].weight)
        target_model.transformer.layers[l][1].fn.ff1.bias.data = make_fxp(
            source_model.transformer.layers[l][1].fn.net[0].bias)
        target_model.transformer.layers[l][1].fn.ff2.weight.data = make_fxp(
            source_model.transformer.layers[l][1].fn.net[3].weight)
        target_model.transformer.layers[l][1].fn.ff2.bias.data = make_fxp(
            source_model.transformer.layers[l][1].fn.net[3].bias)

        target_model.transformer.layers[l][0].fn.to_qkv.weight.data = make_fxp(
            source_model.transformer.layers[l][0].fn.to_qkv.weight)


def save_weights(net):

    with open("../../output/data.cpp", "w") as f:
        def write_infile(param_to_write, param_name):
            param_to_write = param_to_write * (2 ** FRACTION_BITS)
            param_to_write = param_to_write.astype(np.int16)
            f.write("int16_t {}[{}] = ".format(param_name.replace(".", "_"), param_to_write.shape[0]))
            f.write("{")
            for elem in param_to_write:
                f.write(str(elem))
                f.write(", ")
            f.write("};\n")

        for name, param in net.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
                if "to_qkv" in name:
                    q, k, v =  param.detach().transpose(-1, -2).chunk(3, dim=-1)
                    for qkv_mat, qkv_name in zip([q, k, v], ["Q", "K", "V"]):
                        for head in range(4):
                            param_to_write = qkv_mat[:, head*4:(head+1)*4]
                            param_to_write = param_to_write.cpu().numpy().squeeze().reshape(-1,)
                            write_infile(param_to_write, "{}.{}.H{}".format(name, qkv_name, head))
                else:
                    param_to_write = param.detach().cpu().numpy().squeeze()
                    param_to_write = param_to_write.transpose().reshape(-1,) if name != "pos_embedding" \
                        else param_to_write.reshape(-1,)
                    write_infile(param_to_write, name)



def test(opt, test_dataloader, model, print_results=False, target_model = None):
    """
    Test the model trained with the prototypical learning algorithm
    """

    model.eval()

    x_support_set_all, y_support_set_all = get_support_set_per_patient(opt.num_support_val,
                                                               data_dir=opt.siena_data_dir,
                                                               patient_ids=opt.patients)
    prototypes_all = []

    for x_support_set, y_support_set in zip(x_support_set_all, y_support_set_all):
        x_support_set = torch.tensor(x_support_set).to(device)
        y_support_set = torch.tensor(y_support_set).to(device)

        x = x_support_set.reshape((x_support_set.shape[0], 1, -1, x_support_set.shape[3]))
        model_output = model(x)
        prototypes = get_prototypes(model_output, target=y_support_set)
        # print("Prototypes", prototypes)
        prototypes_all.append(prototypes)

    predict = []
    predict_prob = []
    true_label = []
    for batch in tqdm(test_dataloader):
        x, y = batch
        x, y = x.to(device), y.to(device)

        x = x.reshape((x.shape[0], 1, -1, x.shape[3]))

        model_output = model(x)

        prob, output = prototypical_evaluation_per_patient(prototypes_all, model_output)
        predict.append(output.detach().cpu().numpy())
        predict_prob.append(prob.detach().cpu().numpy())
        true_label.append(y.detach().cpu().numpy())
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


def get_trained_model():
    exp_root = os.path.join(options.experiment_root, '_'.join(str(v) for v in options.finetune_patients))
    model_path = os.path.join('../', exp_root, 'last_model.pth')
    model = init_vit(options)
    model.load_state_dict(torch.load(model_path))
    return model


def main():
    model = get_trained_model()
    model.eval()
    l = 0
    net = FixedPointViT(image_size=(3200, 15), patch_size=(80, 5), num_classes=16, dim=16, depth=4, heads=4, mlp_dim=4,
                        pool='cls',
                        channels=1, dim_head=4, dropout=0.2, emb_dropout=0.2)
    net.eval()
    print(model)
    print(net)

    send_weights(model, net)

    save_weights(net)

    all_patients = [0, 1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17]
    test_patient_ids = [p for p in all_patients if p not in options.excluded_patients]
    test_dataloader = get_data_loader_siena(batch_size=32, patient_ids=test_patient_ids,
                                            save_dir=options.siena_data_dir)
    train_dataloader = get_data_loader_siena(batch_size=32, patient_ids=options.patients,
                                             save_dir=options.siena_data_dir)

    # test(options, test_dataloader, model, print_results=True)
    # test(options, test_dataloader, net, print_results=True)
    input_signal = next(iter(test_dataloader))[0].to(device)
    input_signal = input_signal.reshape((input_signal.shape[0], 1, -1, input_signal.shape[3]))

    error = torch.sum(torch.abs(net(input_signal, net.to_patch_embedding_linear.weight.data,
                                    net.to_patch_embedding_linear.bias.data) - model(input_signal)))
    print("Error", error)
    return
    print(total_weights.shape)
    # Create a histogram
    plt.hist(total_weights.detach().cpu().numpy(), bins=20, color='blue', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Total Weights')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
