import numpy as np
import torch
import torch.nn as nn
import os
from few_shot_train import init_vit
import matplotlib.pyplot as plt
from few_shot.prototypical_loss import get_prototypes, prototypical_evaluation, prototypical_evaluation_per_patient
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
import time
from tqdm import tqdm
from few_shot_train import get_support_set_per_patient, init_seed
from dataset import get_data_loader_siena
from utils.parser_util import get_parser
from sklearn.decomposition import PCA


options = get_parser().parse_args()
init_seed(options)
device = 'cuda:0' if torch.cuda.is_available() and options.cuda else 'cpu'

def test(opt, test_dataloader, train_loader, model, print_results=False, target_model = None):
    """
    Test the model trained with the prototypical learning algorithm
    """

    model.eval()

    train_samples_non = np.zeros((0, 16))
    train_samples_seiz = np.zeros((0, 16))
    for batch in tqdm(train_loader):
        x, y = batch
        x, y = x.to(device), y.to(device)

        x = x.reshape((x.shape[0], 1, -1, x.shape[3]))

        model_output = model(x)
        seiz_index = np.where(y.detach().cpu().numpy()==1)[0]
        non_index = np.where(y.detach().cpu().numpy()==0)[0]
        train_samples_seiz = np.concatenate((train_samples_seiz, model_output[seiz_index].detach().cpu().numpy()), axis=0)
        train_samples_non = np.concatenate((train_samples_non, model_output[non_index].detach().cpu().numpy()), axis=0)


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
        # save_signals(prototypes, "prototypes", first_signal=False)
        # print("Prototypes", prototypes)
        prototypes_all.append(prototypes)
        print(prototypes.shape)
        train_samples_seiz = np.concatenate((train_samples_seiz, prototypes[1:2].detach().cpu().numpy()))
        train_samples_non = np.concatenate((train_samples_non, prototypes[0:1].detach().cpu().numpy()))

    tsne = PCA(n_components=2, random_state=42)
    output_2d = tsne.fit_transform(np.concatenate((train_samples_non, train_samples_seiz)))

    train_samples_non_len = train_samples_non.shape[0]
    prototype_len = len(prototypes_all)
    # Separate the transformed model outputs and prototypes
    output_2d_model_non = output_2d[:train_samples_non_len-prototype_len]  # Exclude the last two data points (prototypes)
    output_2d_model_seiz = output_2d[train_samples_non_len:-prototype_len]  # Exclude the last two data points (prototypes)
    output_2d_prototypes_non = output_2d[train_samples_non_len-prototype_len: train_samples_non_len]  # Last two data points are the prototypes
    output_2d_prototypes_seiz = output_2d[-prototype_len:]  # Last two data points are the prototypes

    plt.scatter(output_2d_model_non[:, 0], output_2d_model_non[:, 1], label='Model Outputs', marker ='o', color='green', s=10)
    plt.scatter(output_2d_model_seiz[:, 0], output_2d_model_seiz[:, 1], label='Model Outputs', marker ='o', color='red',  s=10)
    plt.scatter(output_2d_prototypes_non[:, 0], output_2d_prototypes_non[:, 1], label='Prototypes', marker='x', color='green',
                s=200)
    plt.scatter(output_2d_prototypes_seiz[:, 0], output_2d_prototypes_seiz[:, 1], label='Prototypes', marker='x',
                color='red',
                s=200)

    predict = []
    predict_prob = []
    true_label = []
    test_samples_non = np.zeros((0, 16))
    test_samples_seiz = np.zeros((0, 16))
    for batch in tqdm(test_dataloader):
        x, y = batch
        x, y = x.to(device), y.to(device)

        x = x.reshape((x.shape[0], 1, -1, x.shape[3]))

        model_output = model(x)
        seiz_index = np.where(y.detach().cpu().numpy() == 1)[0]
        non_index = np.where(y.detach().cpu().numpy() == 0)[0]
        test_samples_seiz = np.concatenate((test_samples_seiz, model_output[seiz_index].detach().cpu().numpy()),
                                            axis=0)
        test_samples_non = np.concatenate((test_samples_non, model_output[non_index].detach().cpu().numpy()), axis=0)


        prob, output = prototypical_evaluation_per_patient(prototypes_all, model_output)
        predict.append(output.detach().cpu().numpy())
        predict_prob.append(prob.detach().cpu().numpy())
        true_label.append(y.detach().cpu().numpy())
    predict = np.hstack(predict)
    predict_prob = np.hstack(predict_prob)
    true_label = np.hstack(true_label)

    output_2d_seiz = tsne.transform(test_samples_seiz)
    output_2d_non = tsne.transform(test_samples_non)
    # plt.scatter(output_2d_non[:, 0], output_2d_non[:, 1], label='Test', marker='.', color='blue',
    #             s=4)
    plt.scatter(output_2d_seiz[:, 0], output_2d_seiz[:, 1], label='Test', marker='.', color='purple',
                s=4)

    plt.show()

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

    all_patients = [0, 1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17]
    test_patient_ids = [p for p in all_patients if p not in options.excluded_patients]
    test_dataloader = get_data_loader_siena(batch_size=32, patient_ids=test_patient_ids,
                                            save_dir=options.siena_data_dir)
    train_dataloader = get_data_loader_siena(batch_size=32, patient_ids=options.finetune_patients,
                                             save_dir=options.siena_data_dir)

    test(options, test_dataloader, train_dataloader, model, print_results=True)


if __name__ == '__main__':
    main()
