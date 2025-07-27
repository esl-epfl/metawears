# coding=utf-8
import warnings
import os
import numpy as np
import torch
import learn2learn as l2l
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import pandas as pd


# Filter out the specific UserWarning related to torchvision
warnings.filterwarnings("ignore", category=UserWarning, message="Failed to load image Python extension")

from utils.parser_util import get_parser
from dataset import get_data_loader, get_data_loader_siena, get_data_loader_siena_finetune
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

def plot_and_save_metrics(train_loss, val_loss, val_auc, exp_root, prefix=''):
    """
    Plots training & validation loss and validation AUC, then saves the figure.
    """
    epochs = range(1, len(train_loss) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plotting Loss
    ax1.plot(epochs, train_loss, 'bo-', label='Training Loss')
    ax1.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    ax1.set_title(f'{prefix} Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plotting AUC
    ax2.plot(epochs, val_auc, 'go-', label='Validation AUC')
    ax2.set_title(f'{prefix} Validation AUC')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('AUC')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_filename = os.path.join(exp_root, f"{prefix.lower().replace(' ', '_')}_metrics.png")
    plt.savefig(plot_filename)
    print(f"Saved metrics plot to {plot_filename}")
    plt.close()

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write(f"{item}\n")


def adapt_on_patient(opt, meta_lr, fast_lr):
    """
    Fine-tunes (adapts) the meta-trained model on a new patient's data.
    """
    # Create a specific directory for this fine-tuning experiment
    exp_root = os.path.join(opt.experiment_root, '_'.join(str(p) for p in opt.finetune_patients))
    if not os.path.exists(exp_root):
        os.makedirs(exp_root)

    init_seed(opt)

    # 1. Load the fine-tuning and validation data for specific Siena patients
    tr_dataset, tr_label = get_data_loader_siena_finetune(
        patient_ids=opt.finetune_patients,
        save_dir=opt.siena_data_dir
    )
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=opt.classes_per_it_tr * (opt.num_support_tr + opt.num_query_tr), shuffle=True, num_workers=6)

    val_dataloader = get_data_loader_siena(
        batch_size=opt.classes_per_it_val * (opt.num_support_val + opt.num_query_val),
        patient_ids=opt.validation_patients,
        save_dir=opt.siena_data_dir
    )

    # 2. Initialize a new model and load the weights from the meta-trained (base) model
    model = init_vit(opt)
    if not opt.skip_base_learner:
        model_path = os.path.join(opt.base_learner_root, 'best_model.pth')
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded base model from {model_path}")

    # 3. Run the training loop to adapt the model
    print(f"Adapting model on patients: {opt.finetune_patients}")
    train_losses, val_losses, val_aucs = train_maml(
        opt=opt,
        tr_dataloader=tr_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        meta_lr=meta_lr,
        fast_lr=fast_lr,
        exp_root=exp_root  # Save the adapted model in its own directory
    )

    plot_and_save_metrics(train_losses, val_losses, val_aucs, exp_root, prefix=f"Adaptation Patient(s) {options.finetune_patients}")




def evaluate_adapted_model(opt, fast_lr):
    """
    Evaluates a fine-tuned MAML model on unseen patients.
    """
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    loss_func = torch.nn.CrossEntropyLoss()
    
    # 1. Load the specific adapted model for the fine-tuned patients
    model = init_vit(opt)
    exp_root = os.path.join(opt.experiment_root, '_'.join(str(p) for p in opt.finetune_patients))
    model_path = os.path.join(exp_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded adapted model from {model_path} for evaluation.")
    
    maml = l2l.algorithms.MAML(model, lr=fast_lr)
    model.eval()

    # 2. Load the test data from unseen patients
    all_patients = [0, 1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17]
    test_patient_ids = [p for p in all_patients if p not in opt.excluded_patients]
    test_dataloader = get_data_loader_siena(
        batch_size=opt.classes_per_it_val * (opt.num_support_val + opt.num_query_val),
        patient_ids=test_patient_ids,
        save_dir=opt.siena_data_dir
    )
    
    print(f"Evaluating on test patients: {test_patient_ids}")
    
    # 3. Perform evaluation
    true_labels = []
    pred_probs = []

    for batch in tqdm(test_dataloader, desc="Final Evaluation"):
        x_batch, y_batch = batch
        split_point = len(x_batch) // 2
        x_support, y_support = x_batch[:split_point], y_batch[:split_point]
        x_query, y_query = x_batch[split_point:], y_batch[split_point:]

        x_support, y_support = x_support.to(device), y_support.to(device).long()
        x_query, y_query = x_query.to(device), y_query.to(device).long()

        x_support = x_support.reshape((x_support.shape[0], 1, -1, x_support.shape[3]))
        x_query = x_query.reshape((x_query.shape[0], 1, -1, x_query.shape[3]))
        
        learner = maml.clone()
        
        # Adapt on the support set from the test task
        support_preds = learner(x_support)
        support_loss = loss_func(support_preds, y_support)
        learner.adapt(support_loss)
        
        # Evaluate on the query set
        with torch.no_grad():
            query_preds = learner(x_query)
            # Get probabilities for the positive class (seizure) for AUC
            probs = torch.nn.functional.softmax(query_preds, dim=1)[:, 1]
            
            pred_probs.extend(probs.cpu().numpy())
            true_labels.extend(y_query.cpu().numpy())

    auc_score = roc_auc_score(true_labels, pred_probs)
    print(f"Final Test AUC on unseen patients: {auc_score:.4f}")

    # Placeholder for results
    results = {
        "seed": opt.manual_seed,
        "num_support": opt.num_support_val,
        "patients": str(opt.finetune_patients), # Using finetune_patients as the equivalent of the support set
        "finetune_patients": str(opt.finetune_patients),
        "excluded_patients": str(opt.excluded_patients),
        "auc": auc_score,
        "skip_base_learner": False, # MAML script does not perform this ablation
        "skip_finetune": False      # MAML script does not perform this ablation
    }

    # Convert the results dictionary to a DataFrame
    result_df = pd.DataFrame([results])

    # Save results to a CSV file
    output_filename = "../output/results_with_validation.csv"

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
        print(f"Created a new results file: {output_filename}")



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
    val_auc_total = [] 
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
            epoch_true_labels = []
            epoch_pred_probs = []

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

                # Get probabilities for AUC calculation
                probs = torch.nn.functional.softmax(query_preds, dim=1)[:, 1]
                epoch_pred_probs.extend(probs.cpu().detach().numpy())
                epoch_true_labels.extend(y_query.cpu().detach().numpy())
            
            avg_val_loss = meta_val_loss / len(val_dataloader)
            val_loss_total.append(avg_val_loss)

            # Calculate and store validation AUC for the epoch
            epoch_auc = roc_auc_score(epoch_true_labels, epoch_pred_probs)
            val_auc_total.append(epoch_auc)
            
            postfix = ' (Best)' if avg_val_loss < best_val_loss else f' (Best: {best_val_loss:.4f})'
            print(f'Avg Meta-Val Loss: {avg_val_loss:.4f} | Val AUC: {epoch_auc:.4f}{postfix}')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
    
    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss_total', 'val_loss_total']:
        save_list_to_file(os.path.join(exp_root, f"{name}.txt"), locals()[name])

    return train_loss_total, val_loss_total, val_auc_total


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

    train_losses, val_losses, val_aucs = train_maml(
        opt=options,
        tr_dataloader=tr_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        meta_lr=meta_lr,
        fast_lr=fast_lr
    )

    plot_and_save_metrics(train_losses, val_losses, val_aucs, options.experiment_root, prefix='Meta-Training')


if __name__ == '__main__':
    # To run this script, you would execute:
    # python maml_metawears.py --experiment_root ../output/maml --epochs 50 --learning_rate 0.001
    
    # NOTE: You may want to add `meta_lr` and `fast_lr` arguments to your `parser_util.py`
    # for better control.
    
    options = get_parser().parse_args()
    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # MAML specific learning rates
    meta_lr = options.learning_rate
    fast_lr = 0.01

    if options.finetune:
        # This will ONLY run the adaptation step, saving the fine-tuned model.
        print("--- Starting Patient Adaptation ---")
        adapt_on_patient(options, meta_lr=meta_lr * 0.1, fast_lr=fast_lr)

    elif options.eval:
        # This will ONLY run the evaluation on an already fine-tuned model.
        # It requires the same --finetune_patients argument to know which model to load.
        print("--- Starting Evaluation of Adapted Model ---")
        evaluate_adapted_model(options, fast_lr=fast_lr)
        
    else:
        # This is the default action: run the initial meta-training for the base model.
        print("--- Starting Base Model Meta-Training ---")
        main()