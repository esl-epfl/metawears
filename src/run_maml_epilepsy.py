# run_maml_final.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# learn2learn will handle the MAML logic
import learn2learn as l2l

# --- Using YOUR exact parser and utilities ---
from src.utils.parser_util import get_parser
from utils import (
    set_gpu,
    set_seed,
    get_model_dir,
    Logger,
    get_patience,
    EarlyStopping
)
# --- Using YOUR exact data and model loading methods ---
from vit_pytorch.vit import ViT
from dataset import get_data_loader


def main():
    """
    Main function for the MAML experiment, using your project's structure.
    """
    # =========================================================================
    # 1. Setup - Using your parser and utils
    # =========================================================================
    args = get_parser().parse_args()
    set_seed(args.seed)
    set_gpu(args.gpu)
    model_dir = get_model_dir(args)
    logger = Logger(model_dir / f"train_maml_k_{args.k_shot}.log")
    logger.log(f"Running MAML experiment with arguments: {vars(args)}")

    # =========================================================================
    # 2. Data Loading - Using your get_data_loader function
    # =========================================================================
    train_dataset, _, _, _, _, _ = get_data_loader(
        batch_size=args.batch_size, # Your loader takes batch_size
        save_dir=args.data_path # Pass the data path to your loader
    )
    logger.log("Data loaded successfully using your get_data_loader.")

    # Now, we wrap your dataset with learn2learn to create meta-tasks
    meta_train_dataset = l2l.data.MetaDataset(train_dataset)

    train_tasks = l2l.data.TaskDataset(
        meta_train_dataset,
        task_transforms=[
            # These transforms create the N-way, K-shot tasks for meta-learning
            l2l.data.transforms.NWays(meta_train_dataset, n=args.n_way),
            l2l.data.transforms.KShots(meta_train_dataset, k=args.k_shot + args.k_query),
            l2l.data.transforms.LoadData(meta_train_dataset),
            l2l.data.transforms.RemapLabels(meta_train_dataset), # Optional, but good practice
            l2l.data.transforms.ConsecutiveLabels(meta_train_dataset), # Optional
        ],
        num_tasks=args.meta_train_iterations * args.meta_batch_size,
    )

    # Finally, create the DataLoader that the training loop will use
    train_loader = torch.utils.data.DataLoader(train_tasks, pin_memory=True, batch_size=args.meta_batch_size)
    logger.log("Wrapped dataset with learn2learn for meta-task generation.")



    # =========================================================================
    # 3. Model & MAML Setup
    # =========================================================================
    # Initialize Vision Transformer exactly as in your other scripts
    model = VisionTransformer(
        patch_size=args.patch_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        num_classes=args.n_way, # N-way classification for meta-learning
        in_channels=args.in_channels,
        dropout=args.dropout,
    ).cuda()

    # Wrap the model with the learn2learn MAML algorithm
    maml = l2l.algorithms.MAML(
        model,
        lr=args.fast_lr,
        first_order=False
    )

    # Setup the meta-optimizer and loss function
    meta_optimizer = optim.Adam(maml.parameters(), lr=args.meta_lr)
    loss_fn = nn.CrossEntropyLoss()
    logger.log("Model initialized and wrapped with learn2learn MAML.")

    # =========================================================================
    # 4. Meta-Training Loop
    # =========================================================================
    logger.log("Starting meta-training...")
    for iteration in tqdm(range(args.meta_train_iterations), desc="Meta-Training"):
        meta_train_loss = 0.0
        meta_train_accuracy = 0.0

        meta_optimizer.zero_grad()

        # Your data loader provides a batch of tasks. We process them one by one.
        # This loop aggregates gradients for one meta-update.
        for i, batch in enumerate(train_loader):
            if i >= args.meta_batch_size:
                break
            
            learner = maml.clone()
            support_data, support_labels, query_data, query_labels = [d.cuda() for d in batch]

            # Inner Loop: Adapt on the support set
            for _ in range(args.num_inner_updates):
                preds = learner(support_data)
                loss = loss_fn(preds, support_labels)
                learner.adapt(loss)

            # Outer Loop: Compute loss on the query set
            query_preds = learner(query_data)
            query_loss = loss_fn(query_preds, query_labels)
            meta_train_loss += query_loss

            with torch.no_grad():
                query_accuracy = (query_preds.argmax(dim=1) == query_labels).float().mean()
                meta_train_accuracy += query_accuracy

        # Average loss and backpropagate for the meta-update
        meta_train_loss /= args.meta_batch_size
        meta_train_accuracy /= args.meta_batch_size
        meta_train_loss.backward()
        meta_optimizer.step()

        if iteration % args.log_interval == 0:
            logger.log(f"Iteration {iteration:04d} | "
                       f"Meta-Train Loss: {meta_train_loss.item():.4f} | "
                       f"Meta-Train Accuracy: {meta_train_accuracy.item():.4f}")

    logger.log("Meta-training finished.")

    # =========================================================================
    # 5. Save Final Model
    # =========================================================================
    model_path = model_dir / f"maml_vit_final_k{args.k_shot}.pth"
    torch.save(model.state_dict(), model_path)
    logger.log(f"Meta-trained model saved to {model_path}")
    logger.log("Next step: Implement the meta-testing phase.")


if __name__ == "__main__":
    main()