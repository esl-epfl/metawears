import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
from copy import deepcopy
from vit_learner import VitLearner
from sklearn.metrics import roc_auc_score

class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = VitLearner()
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, setsz, 1, freq_channel, time_dim]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, 1, freq_channel, time_dim]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, ch, freq_channel, time_dim = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0.0 for _ in range(self.update_step + 1)]

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits, _ = self.net(x_spt[i], vars=None)
            logits = logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_spt[i])
            # compute the grad and update theta parameters with the gradients
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q, _ = self.net(x_qry[i], self.net.parameters())
                del _
                logits_q = logits_q.squeeze(-1)
                loss_q = F.binary_cross_entropy_with_logits(logits_q, y_qry[i])
                losses_q[0] += loss_q

                predict_prob = F.sigmoid(logits_q).clone().detach().cpu().numpy()
                true_label = y_qry[i].clone().detach().cpu().numpy()
                auc = roc_auc_score(true_label, predict_prob)
                corrects[0] = corrects[0] + auc

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]

                logits_q, _ = self.net(x_qry[i], fast_weights)
                del _
                logits_q = logits_q.squeeze(-1)
                loss_q = F.binary_cross_entropy_with_logits(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                predict_prob = F.sigmoid(logits_q).clone().detach().cpu().numpy()
                true_label = y_qry[i].clone().detach().cpu().numpy()
                auc = roc_auc_score(true_label, predict_prob)
                corrects[1] = corrects[1] + auc

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits, new_model = self.net(x_spt[i], fast_weights)
                logits = logits.squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y_spt[i])
                # 2. compute grad on theta_pi

                grad = torch.autograd.grad(loss, new_model.parameters(), create_graph=True, materialize_grads=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights)
                logits_q = logits_q.squeeze(-1)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.binary_cross_entropy_with_logits(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    predict_prob = F.sigmoid(logits_q).clone().detach().cpu().numpy()
                    true_label = y_qry[i].clone().detach().cpu().numpy()
                    auc = roc_auc_score(true_label, predict_prob)
                    corrects[k + 1] = corrects[k + 1] + auc

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        print('meta update')
        for p in self.net.parameters()[:5]:
            print(torch.norm(p).item())
        self.meta_optim.step()

        accs = np.array(corrects) / (task_num)

        return accs

    def get_model_copy(self, fast_weight):
        new_model = deepcopy(self.net)
        for (value), (value_fw) in zip(new_model.parameters(), fast_weight):
            value.data = value_fw.data
        return new_model

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters())
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        del net

        accs = np.array(corrects) / querysz

        return accs


def main():
    pass


if __name__ == '__main__':
    main()
