# coding=utf-8
import warnings
import os
import numpy as np
import torch
import learn2learn as l2l
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score

# Filter out the specific UserWarning related to torchvision
warnings.filterwarnings("ignore", category=UserWarning, message="Failed to load image Python extension")

from utils.parser_util import get_parser
from dataset import get_data_loader
from vit_pytorch.vit import ViT

def init_seed(opt):
    """
    Disable cudnn to maximize reproducibility.
    """
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

def init_vit(opt):
    """
    Initialize the Vision Transformer model.
    """
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ViT(
        image_size=(3200, 15),
        patch_size=(80, 5),
        num_classes=2,  # Binary classification: seizure or non-seizure
        dim=16,
        depth=4,
        heads=4,
        mlp_dim=4,
        pool='cls',
        channels=1,
        dim_head=4,
        dropout=0.2,
        emb_dropout=0.2
    ).to(device)
    return model

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write(f"{item}\n")

def train_maml(opt, tr_dataloader, model, meta_lr, fast_lr, val_dataloader=None, exp_root=None):
    """
    Train the model with the MAML algorithm.
    """
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    
    # Wrap the model with MAML
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    optimizer = torch.optim.Adam(maml.parameters(), lr=meta_lr)
    loss_func = torch.nn.CrossEntropyLoss()

    if exp_root is None:
        exp_root = opt.experiment_root
    
    best_model_path = os.path.join(exp_root, 'best_model.pth')
    last_model_path = os.path.join(exp_root, 'last_model.pth')
    
    train_loss_total = []
    val_loss_total = []
    best_val_loss = float('inf')

    for epoch in range(opt.epochs):
        print(f'=== Epoch: {epoch} ===')
        meta_train_loss = 0.0
        
        # Using a generic iterator since PrototypicalBatchSampler is not needed
        tr_iter = iter(tr_dataloader)
        model.train()

        for batch in tqdm(tr_iter, total=opt.iterations):
            optimizer.zero_grad()
            
            x_batch, y_batch = batch
            # MAML needs a support/query split for each task
            # Here, we treat each batch as a single "task" for simplicity.
            # A more advanced setup might use a MetaDataset.
            
            # Splitting data into support and query sets
            split_point = len(x_batch) // 2
            x_support, y_support = x_batch[:split_point], y_batch[:split_point]
            x_query, y_query = x_batch[split_point:], y_batch[split_point:]

            x_support, y_support = x_support.to(device), y_support.to(device).long()
            x_query, y_query = x_query.to(device), y_query.to(device).long()
            
            # Reshape for ViT
            x_support = x_support.reshape((x_support.shape[0], 1, -1, x_support.shape[3]))
            x_query = x_query.reshape((x_query.shape[0], 1, -1, x_query.shape[3]))
            
            learner = maml.clone()
            
            # 1. Adapt the model
            # For simplicity, one adaptation step. Can be configured.
            support_preds = learner(x_support)
            support_loss = loss_func(support_preds, y_support)
            learner.adapt(support_loss)
            
            # 2. Evaluate the adapted model
            query_preds = learner(x_query)
            query_loss = loss_func(query_preds, y_query)
            
            # 3. Meta-update
            query_loss.backward()
            for p in maml.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(1.0 / opt.iterations)
            optimizer.step()
            
            meta_train_loss += query_loss.item()
        
        avg_train_loss = meta_train_loss / opt.iterations
        train_loss_total.append(avg_train_loss)
        print(f'Avg Meta-Train Loss: {avg_train_loss}')

        # Validation phase
        if val_dataloader:
            model.eval()
            meta_val_loss = 0.0
            val_iter = iter(val_dataloader)
            
            for batch in tqdm(val_iter, desc="Validation"):
                x_batch, y_batch = batch
                x_support, y_support = x_batch[:split_point], y_batch[:split_point]
                x_query, y_query = x_batch[split_point:], y_batch[split_point:]

                x_support, y_support = x_support.to(device), y_support.to(device).long()
                x_query, y_query = x_query.to(device), y_query.to(device).long()
                
                x_support = x_support.reshape((x_support.shape[0], 1, -1, x_support.shape[3]))
                x_query = x_query.reshape((x_query.shape[0], 1, -1, x_query.shape[3]))

                learner = maml.clone()
                support_preds = learner(x_support)
                support_loss = loss_func(support_preds, y_support)
                learner.adapt(support_loss)
                
                query_preds = learner(x_query)
                query_loss = loss_func(query_preds, y_query)
                meta_val_loss += query_loss.item()
            
            avg_val_loss = meta_val_loss / len(val_dataloader)
            val_loss_total.append(avg_val_loss)
            
            postfix = ' (Best)' if avg_val_loss < best_val_loss else f' (Best: {best_val_loss:.4f})'
            print(f'Avg Meta-Val Loss: {avg_val_loss:.4f}{postfix}')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
    
    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss_total', 'val_loss_total']:
        save_list_to_file(os.path.join(exp_root, f"{name}.txt"), locals()[name])

def main():
    """
    Initialize and meta-train the MAML model.
    """
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    init_seed(options)
    
    # MAML works better with standard dataloaders
    # We remove the Prototypical Sampler
    tr_dataset, val_dataset, _, tr_label, val_label, _ = get_data_loader(
        batch_size=options.num_support_tr + options.num_query_tr
    )
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=options.classes_per_it_tr * (options.num_support_tr + options.num_query_tr), shuffle=True, num_workers=6)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=options.classes_per_it_val * (options.num_support_val + options.num_query_val), shuffle=False, num_workers=6)

    model = init_vit(options)
    
    # Add MAML specific learning rates to your parser or define them here
    meta_lr = options.learning_rate
    fast_lr = 0.01 # Example fast learning rate for inner loop

    train_maml(
        opt=options,
        tr_dataloader=tr_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        meta_lr=meta_lr,
        fast_lr=fast_lr
    )

if __name__ == '__main__':
    # To run this script, you would execute:
    # python maml_metawears.py --experiment_root ../output/maml --epochs 50 --learning_rate 0.001
    
    # NOTE: You may want to add `meta_lr` and `fast_lr` arguments to your `parser_util.py`
    # for better control.
    
    options = get_parser().parse_args()
    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    # For now, this script only demonstrates the meta-training part.
    # The finetuning and evaluation loops would follow a similar logic to the validation loop,
    # adapting the loaded meta-trained model on the support set of the new patient
    # before making predictions on the query set.
    main()