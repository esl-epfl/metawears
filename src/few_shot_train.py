# coding=utf-8
import json
import time
import warnings

# Filter out the specific UserWarning related to torchvision
warnings.filterwarnings("ignore", category=UserWarning, message="Failed to load image Python extension")
# TODO solve the CUDA version issue

from few_shot.prototypical_batch_sampler import PrototypicalBatchSampler
from few_shot.prototypical_loss import prototypical_loss as loss_fn
from few_shot.prototypical_loss import get_prototypes, prototypical_evaluation, prototypical_evaluation_per_patient
from utils.parser_util import get_parser
from dataset import get_data_loader, get_data_loader_siena, get_data_loader_siena_finetune
from few_shot.support_set_const import seizure_support_set, non_seizure_support_set
from few_shot.support_set_const import seizure_support_set_siena, non_seizure_support_set_siena
from utils.utils import thresh_max_f1
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import os
from vit_pytorch.vit import ViT
import pickle
from matplotlib import pyplot as plt


def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt):
    ret = get_data_loader(batch_size=2 * (opt.num_support_tr + opt.num_query_tr), save_dir=opt.TUSZ_data_dir)
    tr_dataset, val_dataset, test_dataset, tr_label, val_label, test_label = ret

    tr_sampler = init_sampler(opt, tr_label, mode="train")
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_sampler=tr_sampler, num_workers=6)
    val_sampler = init_sampler(opt, val_label, mode="val")
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=6)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, num_workers=6)

    return tr_dataloader, val_dataloader, test_dataloader


def init_vit(opt):
    """
    Initialize the ViT
    """
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ViT(image_size=(3200, 15), patch_size=(80, 5), num_classes=16, dim=16, depth=4, heads=4, mlp_dim=4,
                pool='cls',
                channels=1, dim_head=4, dropout=0.2, emb_dropout=0.2).to(device)
    return model


def init_optim(opt, model, ratio=1.0):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate * ratio)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def plot_and_save_curves(train_loss, val_loss, train_acc, val_acc, exp_root):
    """
    Plots training & validation loss and accuracy curves, then saves the figure.
    """
    epochs = range(1, len(train_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot Loss Curve ---
    ax1.plot(epochs, train_loss, 'bo-', label='Training Loss')
    ax1.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # --- Plot Accuracy Curve ---
    ax2.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
    ax2.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    # Save the figure
    plot_path = os.path.join(exp_root, "training_curves.png")
    plt.savefig(plot_path)
    print(f"📈 Saved training curves to {plot_path}")
    plt.close()


def get_support_set(n_sample, data_dir):
    support_set = []
    labels = []
    for label, class_support_set in enumerate([non_seizure_support_set, seizure_support_set]):
        for filename in np.random.permutation(class_support_set)[:n_sample]:
            filepath = os.path.join(data_dir + "/task-binary_datatype-train_STFT/",
                                    filename + ".pkl")
            with open(filepath, 'rb') as f:
                data_pkl = pickle.load(f)
                signals = np.asarray(data_pkl['STFT'])
                support_set.append(signals)
                labels.append(label)

    return np.array(support_set), np.array(labels)


def get_support_set_per_patient(n_sample, data_dir, patient_ids):
    support_set = []
    labels = []
    file_dir = os.path.join(data_dir, 'task-binary_datatype-eval_STFT')
    for pat_id in patient_ids:
        support_set_patient = []
        labels_patient = []
        file_lists = {'bckg': [], 'seiz': []}

        filenames = os.listdir(file_dir)
        for filename in filenames:
            patient = int(filename[2:4])
            if patient!=pat_id:
                continue
            if 'bckg' in filename:
                file_lists['bckg'].append(os.path.join(file_dir, filename))
            elif 'seiz' in filename:
                file_lists['seiz'].append(os.path.join(file_dir, filename))
            else:
                print('------------------------  error  ------------------------')
                exit(-1)


        for label, class_support_set in enumerate([file_lists['bckg'], file_lists['seiz']]):
            for filename in np.random.permutation(class_support_set)[:n_sample]:
                # filepath = os.path.join(data_dir + "/task-binary_datatype-eval_STFT/",
                #                         filename)
                filepath = filename
                with open(filepath, 'rb') as f:
                    data_pkl = pickle.load(f)
                    signals = np.asarray(data_pkl['STFT'])
                    support_set_patient.append(signals)
                    labels_patient.append(label)

        support_set.append(np.array(support_set_patient))
        labels.append(np.array(labels_patient))

    return support_set, labels


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None, exp_root=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    if val_dataloader is None:
        best_state = None

    train_loss_total = []
    train_acc_total = []
    val_loss_total = []
    val_acc_total = []
    best_acc = 0
    best_auc = 0

    if exp_root is None:
        exp_root = opt.experiment_root
    model_path = os.path.join(exp_root)
    best_model_path = os.path.join(exp_root, 'best_model.pth')
    last_model_path = os.path.join(exp_root, 'last_model.pth')

    for epoch in range(opt.epochs):
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        # x_support_set, y_support_set = get_support_set(opt.num_support_tr, opt.TUSZ_data_dir)
        # x_support_set = torch.tensor(x_support_set).to(device)
        # y_support_set = torch.tensor(y_support_set).to(device)
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x_batch, y_batch = batch
            x_support_set = torch.concatenate((x_batch[:opt.num_support_tr], x_batch[2*opt.num_support_tr:3*opt.num_support_tr]))
            y_support_set = torch.concatenate((y_batch[:opt.num_support_tr], y_batch[2*opt.num_support_tr:3*opt.num_support_tr]))

            x_query_set = torch.concatenate((x_batch[opt.num_support_tr:2*opt.num_support_tr], x_batch[3*opt.num_support_tr:]))
            y_query_set =torch.concatenate((y_batch[opt.num_support_tr:2*opt.num_support_tr], y_batch[3*opt.num_support_tr:]))

            x_support_set = x_support_set.clone().detach().to(device)
            y_support_set = y_support_set.clone().detach().to(device)
            x_query_set, y_query_set = x_query_set.to(device), y_query_set.to(device)

            x = torch.concatenate((x_support_set, x_query_set))
            y = torch.concatenate((y_support_set, y_query_set))

            x = x.reshape((x.shape[0], 1, -1, x.shape[3]))

            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-opt.iterations:])
        train_loss_total.append(avg_loss)
        avg_acc = np.mean(train_acc[-opt.iterations:])
        train_acc_total.append(avg_acc)
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()

        # torch.save(model.state_dict(), os.path.join(model_path, "epoch{}.pth".format(epoch)))
        if val_dataloader is None:
            continue
        auc = test(opt, val_dataloader, model, print_results=False, return_results=True)
        postfix = ' (Best)' if auc >= best_auc else ' (Best: {})'.format(
            best_auc)
        print('Val auc: {}{}'.format(auc, postfix))
        if auc >= best_auc:
            torch.save(model.state_dict(), best_model_path)
            best_auc = auc
            best_state = model.state_dict()
        continue

        
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in tqdm(val_iter):
            x_batch, y_batch = batch
            x_support_set = torch.concatenate(
                (x_batch[:opt.num_support_tr], x_batch[2 * opt.num_support_tr:3 * opt.num_support_tr]))
            y_support_set = torch.concatenate(
                (y_batch[:opt.num_support_tr], y_batch[2 * opt.num_support_tr:3 * opt.num_support_tr]))

            x_query_set = torch.concatenate(
                (x_batch[opt.num_support_tr:2 * opt.num_support_tr], x_batch[3 * opt.num_support_tr:]))
            y_query_set = torch.concatenate(
                (y_batch[opt.num_support_tr:2 * opt.num_support_tr], y_batch[3 * opt.num_support_tr:]))

            x_support_set = x_support_set.clone().detach().to(device)
            y_support_set = y_support_set.clone().detach().to(device)
            x_query_set, y_query_set = x_query_set.to(device), y_query_set.to(device)

            x = torch.concatenate((x_support_set, x_query_set))
            y = torch.concatenate((y_support_set, y_query_set))

            x = x.reshape((x.shape[0], 1, -1, x.shape[3]))

            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-opt.iterations:])
        val_loss_total.append(avg_loss)
        avg_acc = np.mean(val_acc[-opt.iterations:])
        val_acc_total.append(avg_acc)
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss_total', 'train_acc_total', 'val_loss_total', 'val_acc_total']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])
        
    # plot_and_save_curves(train_loss_total, val_loss_total, train_acc_total, val_acc_total, exp_root)


    return best_state, best_acc, train_loss_total, train_acc_total, val_loss_total, val_acc_total


def test(opt, test_dataloader, model, print_results=False, return_results=False):
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

    for x_support_set, y_support_set in zip(x_support_set_all, y_support_set_all):
        x_support_set = torch.tensor(x_support_set).to(device)
        y_support_set = torch.tensor(y_support_set).to(device)

        x = x_support_set.reshape((x_support_set.shape[0], 1, -1, x_support_set.shape[3]))
        model_output = model(x)

        prototypes = get_prototypes(model_output, target=y_support_set)
        prototypes_all.append(prototypes)

    predict = []
    predict_prob = []
    true_label = []
    for batch in tqdm(test_dataloader):
        x, y = batch
        x, y = x.to(device), y.to(device)

        x = x.reshape((x.shape[0], 1, -1, x.shape[3]))
        model_output = model(x)
        prob, output, _ = prototypical_evaluation_per_patient(prototypes_all, model_output)
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
        return

    if return_results:
        return roc_auc_score(true_label, predict_prob)

    # Convert the results dictionary to a DataFrame
    result_df = pd.DataFrame([results])

    # Save results to a JSON file
    # output_filename = "../output/results/results_with_validation.csv"
    output_filename = os.path.join(opt.experiment_root, 'results.csv')

    # Check if the file exists
    try:
        # Read the existing file
        existing_df = pd.read_csv(output_filename)

        # Concatenate the new results to the existing DataFrame
        updated_df = pd.concat([existing_df, result_df], ignore_index=True)

        # Write the updated DataFrame back to the file
        updated_df.to_csv(output_filename, index=False)
    except FileNotFoundError:
        # If the file doesn't exist, create a new file with the DataFrame
        result_df.to_csv(output_filename, index=False)
        print("Created a new file")


def eval(options):
    """
    Initialize everything and train
    """

    init_seed(options)
    all_patients = [0, 1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17]
    test_patient_ids = [p for p in all_patients if p not in options.excluded_patients]
    test_dataloader = get_data_loader_siena(batch_size=options.num_query_val,
                                            patient_ids=test_patient_ids,
                                            save_dir=options.siena_data_dir)
    model = init_vit(options)
    if options.skip_finetune:
        exp_root = options.base_learner_root
        model_path = os.path.join(exp_root, 'best_model.pth')
    else:
        exp_root = os.path.join(options.experiment_root, '_'.join(str(v) for v in options.finetune_patients))
        model_path = os.path.join(exp_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         model=model,
         print_results=False)


def main():
    """
    Initialize everything and train
    """
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    tr_dataloader, val_dataloader, test_dataloader = init_dataloader(options)

    model = init_vit(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res


def finetune(options):
    """
    Initialize everything and train
    """
    exp_root = os.path.join(options.experiment_root, '_'.join(str(v) for v in options.finetune_patients))
    if not os.path.exists(exp_root):
        os.makedirs(exp_root)

    init_seed(options)
    tr_dataset, tr_label = get_data_loader_siena_finetune(patient_ids=options.finetune_patients,
                                            save_dir=options.siena_data_dir)

    tr_sampler = init_sampler(options, tr_label, mode="train")
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_sampler=tr_sampler, num_workers=6)

    val_dataloader = get_data_loader_siena(batch_size=options.num_query_val,
                                           patient_ids=options.validation_patients,
                                           save_dir=options.siena_data_dir)

    model = init_vit(options)
    if not options.skip_base_learner:
        model_path = os.path.join(options.base_learner_root, 'best_model.pth')
        model.load_state_dict(torch.load(model_path))
        optim = init_optim(options, model, ratio=0.02)
    else:
        optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    train(opt=options,
          tr_dataloader=tr_dataloader,
          val_dataloader=val_dataloader,
          model=model,
          optim=optim,
          lr_scheduler=lr_scheduler,
          exp_root=exp_root)


if __name__ == '__main__':
    options = get_parser().parse_args()
    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if options.finetune:
        finetune(options)
    elif options.eval:
        eval(options)
    elif options.train:
        main()
    else:
        print("One options should be chosen.")
