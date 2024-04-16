import torch, os
import numpy as np
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random, sys, pickle
import argparse
from tqdm import tqdm
from src.dataset import get_data_loader
from meta import Meta
from few_shot.prototypical_batch_sampler import PrototypicalBatchSampler


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
    ret = get_data_loader(batch_size=2 * (opt.num_support_tr + opt.num_query_tr))
    tr_dataset, val_dataset, test_dataset, tr_label, val_label, test_label = ret

    tr_sampler = init_sampler(opt, tr_label, mode="train")
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_sampler=tr_sampler, num_workers=6)
    val_sampler = init_sampler(opt, val_label, mode="val")
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=6)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, num_workers=6)

    return tr_dataloader, val_dataloader, test_dataloader


def main():
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    device = torch.device('cuda')
    maml = Meta(args).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    tr_dataloader, val_dataloader, test_dataloader = init_dataloader(args)

    for epoch in range(args.epoch):
        # fetch meta_batchsz num of episode each time
        x_support_set_stack = []
        x_query_set_stack = []
        y_support_set_stack = []
        y_query_set_stack = []
        tr_iter = iter(tr_dataloader)
        for step, batch in tqdm(enumerate(tr_iter)):
            x_batch, y_batch = batch
            x_support_set = torch.concatenate(
                (x_batch[:args.num_support_tr], x_batch[2 * args.num_support_tr:3 * args.num_support_tr]))
            y_support_set = torch.concatenate(
                (y_batch[:args.num_support_tr], y_batch[2 * args.num_support_tr:3 * args.num_support_tr]))

            x_query_set = torch.concatenate(
                (x_batch[args.num_support_tr:2 * args.num_support_tr], x_batch[3 * args.num_support_tr:]))
            y_query_set = torch.concatenate(
                (y_batch[args.num_support_tr:2 * args.num_support_tr], y_batch[3 * args.num_support_tr:]))

            x_support_set = x_support_set.clone().detach().to(device)
            y_support_set = y_support_set.clone().detach().to(device)
            x_query_set, y_query_set = x_query_set.to(device), y_query_set.to(device)

            x_support_set = x_support_set.reshape((x_support_set.shape[0], 1, -1, x_support_set.shape[3]))
            x_query_set = x_query_set.reshape((x_query_set.shape[0], 1, -1, x_query_set.shape[3]))

            x_support_set_stack.append(x_support_set)
            x_query_set_stack.append(x_query_set)
            y_support_set_stack.append(y_support_set)
            y_query_set_stack.append(y_query_set)

        x_support_set = torch.stack(x_support_set_stack)
        x_query_set = torch.stack(x_query_set_stack)
        y_support_set = torch.stack(y_support_set_stack)
        y_query_set = torch.stack(y_query_set_stack)
        print("Shapes: ", x_support_set.shape, x_query_set.shape, y_support_set.shape, y_query_set.shape)
        # accs = maml(x_support_set, y_support_set, x_query_set, y_query_set)

        # if epoch % 30 == 0:
        #     print('step:', epoch, '\ttraining acc:', accs)

            # if step % 500 == 0:  # evaluation
            #     db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
            #     accs_all_test = []
            #
            #     for x_spt, y_spt, x_qry, y_qry in db_test:
            #         x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
            #             x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
            #
            #         accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
            #         accs_all_test.append(accs)
            #
            #     # [b, update_step+1]
            #     accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
            #     print('Test acc:', accs)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=100)

    argparser.add_argument('-cTr', '--classes_per_it_tr',
                        type=int,
                        help='number of random classes per episode for training, default=60',
                        default=2)

    argparser.add_argument('-nsTr', '--num_support_tr',
                        type=int,
                        help='number of samples per class to use as support for training, default=5',
                        default=5)

    argparser.add_argument('-nqTr', '--num_query_tr',
                        type=int,
                        help='number of samples per class to use as query for training, default=5',
                        default=5)

    argparser.add_argument('-cVa', '--classes_per_it_val',
                        type=int,
                        help='number of random classes per episode for validation, default=5',
                        default=2)

    argparser.add_argument('-nsVa', '--num_support_val',
                        type=int,
                        help='number of samples per class to use as support for validation, default=5',
                        default=5)

    argparser.add_argument('-nqVa', '--num_query_val',
                        type=int,
                        help='number of samples per class to use as query for validation, default=15',
                        default=1024)

    args = argparser.parse_args()

    main()
