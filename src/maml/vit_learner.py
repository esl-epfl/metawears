import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
from vit_pytorch.vit import ViT


class VitLearner(nn.Module):
    """

    """

    def __init__(self):
        """
        Initialize the ViT
        """
        super(VitLearner, self).__init__()
        model = ViT(image_size=(3200, 15), patch_size=(80, 5), num_classes=16, dim=16, depth=4, heads=4, mlp_dim=4,
                    pool='cls', channels=1, dim_head=4, dropout=0.2, emb_dropout=0.2)
        self.model = model
        self.vars = []
        # save a list as all the parameters that need to be optimized
        self.var_names = []
        for name, param in model.named_parameters():
            # If its requires_grad is True, then append the parameter to the list
            if param.requires_grad:
                self.var_names.append(name)
                self.vars.append(param)

    def forward(self, x, vars=None):
        """
        :param x: [b, 1, ..., ...]
        :param vars:dict of named parameters
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            model = self.model
        else:
            model = ViT(image_size=(3200, 15), patch_size=(80, 5), num_classes=16, dim=16, depth=4, heads=4, mlp_dim=4,
                        pool='cls', channels=1, dim_head=4, dropout=0.2, emb_dropout=0.2)
            # load the vars as parameters of the model based on the var_names
            for name, param in model.named_parameters():
                if name in self.var_names:
                    param.data = vars[name]

        x = model(x)
        return x

    def parameters(self, **kwargs):
        """
        override this function since initial parameters will return with a generator.
        :return:

        Parameters
        ----------
        **kwargs
        """
        return self.vars

    def parameter_names(self):
        """
        return the parameter names
        """
        return self.var_names
