import argparse
import pickle
from copy import deepcopy
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import fewshot as fs
from training import TRAIN_DIRECTORY
from utilities import set_seed

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def train(args):
    """
    Main training function for few-shot learning using an episodic setup.
    """
    episodic_loader = _create_episodic_dataloader(
        fraction=args.fraction,
        n_way=args.num_classes,
        k_shot=args.k_shot,
        q_query=args.q_query
    )

    model = fs.EfficientMetaDualFusionNet(
        num_classes=args.num_classes,
        cnn_channels=args.cnn_channels,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        patch_size=args.patch_size
    ).to(DEVICE)

    optimizer = _get_optimizer(
        model.parameters(),
        lr=args.lr,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay
    )

    criterion = nn.CrossEntropyLoss()
    set_seed()

    _train_multiple_runs(
        n_trainings=args.num_trainings,
        model=model,
        model_name=args.model_name,
        episodic_loader=episodic_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        n_way=args.num_classes,
        k_shot=args.k_shot,
        q_query=args.q_query,
        num_episodes=args.num_episodes
    )


def _train_multiple_runs(
    n_trainings,
    model,
    model_name,
    episodic_loader,
    optimizer,
    criterion,
    device,
    n_way,
    k_shot,
    q_query,
    num_episodes
):
    """
    Runs few-shot training multiple times to produce multiple model instances.
    Saves each model state_dict and records the training history.
    """
    losses_dict = {}
    for i in range(n_trainings):
        run_id = f"{model_name}_{i+1}"
        print(f"\n[INFO] Training run {i+1} of {n_trainings} (model: {run_id})")

        # Clone model and optimizer states so each training run is independent
        model_clone = deepcopy(model)
        optimizer_clone = deepcopy(optimizer)

        # Run episodic few-shot training
        losses = fs.train_few_shot(
            model_clone,
            episodic_loader,
            optimizer_clone,
            criterion,
            device,
            n_way,
            k_shot,
            q_query,
            num_episodes
        )

        # Record training history and save model weights
        losses_dict[run_id] = losses
        torch.save(model_clone.state_dict(), f"{run_id}.pth")

    # Save the losses from all runs to a single pkl file
    with open(f"{model_name}_training_histories.pkl", "wb") as handle:
        pickle.dump(losses_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _create_episodic_dataloader(fraction, n_way, k_shot, q_query):
    """
    Creates an episodic DataLoader for a fraction of the TRAIN_DIRECTORY dataset.
    
    1. Computes mean/std from a subset of data.
    2. Applies base transforms for support samples and augmented transforms for queries.
    3. Returns a DataLoader that yields few-shot episodes.
    """
    # Creating an un-augmented version of the dataset to get its mean and std
    base_transform = fs._get_base_transform()
    dataset_tmp = datasets.ImageFolder(TRAIN_DIRECTORY, transform=base_transform) # No transforms, used for mean and std calculations 
    dataset_indices = fs.create_balanced_subset(dataset_tmp, fraction=fraction)
    mean, std = fs.compute_mean_std(Subset(dataset_tmp, dataset_indices))

    # Applying this mean and std to the transforms for actual training
    noaug_transform = fs._get_base_transform(mean=mean, std=std, normalized=True)
    aug_transform = fs._get_augmented_transform(mean=mean, std=std)

    # Creating separate datasets: augmented for queries and non-augmented for supports
    full_dataset = datasets.ImageFolder(TRAIN_DIRECTORY, transform=aug_transform)
    full_dataset_noaug = datasets.ImageFolder(TRAIN_DIRECTORY, transform=noaug_transform)

    dataset = Subset(full_dataset, dataset_indices)
    dataset_noaug = Subset(full_dataset_noaug, dataset_indices)

    # Episodic Dataset contains both, each training task involves queries and supports
    episodic_dataset = fs.EpisodicDataset(dataset, dataset_noaug, n_way, k_shot, q_query)
    
    return DataLoader(episodic_dataset, batch_size=1, shuffle=True)

def _get_optimizer(parameters, lr=0.001, optimizer_name='adam', **kwargs):
    """
    Returns a PyTorch optimizer given its name as a string.
    
    Args:
        parameters (iterable): Model parameters to optimize.
        lr (float): Learning rate.
        optimizer_name (str): Name of the optimizer (e.g., 'adam', 'sgd').
        **kwargs: Additional keyword arguments for the optimizer.
    """
    optimizer_name = optimizer_name.lower()
    
    optimizers = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'adagrad': optim.Adagrad,
        'rmsprop': optim.RMSprop,
        'adamax': optim.Adamax,
        'asgd': optim.ASGD,
        'nadam': optim.NAdam,
        'lbfgs': optim.LBFGS
    }
    
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizers[optimizer_name](parameters, lr=lr, **kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Few-shot learning model arguments")

    # Model configuration
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes (n-way)')
    parser.add_argument('--cnn_channels', type=int, default=32, help='Number of CNN output channels')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size for Vision Transformer')

    # Data setup
    parser.add_argument('--k_shot', type=int, default=5, help='Number of shots (examples per class)')
    parser.add_argument('--q_query', type=int, default=15, help='Number of query samples per class')
    parser.add_argument('--fraction', type=float, default=0.01, help='Fraction of the dataset to use')

    # Training setup
    parser.add_argument('--num_trainings', type=int, default=3, help='Number of training runs')
    parser.add_argument('--num_episodes', type=int, default=3000, help='Number of episodes per training run')
    parser.add_argument('--model_name', type=str, default='tmp', help='Model name identifier')

    # Optimization
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer name')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')

    args = parser.parse_args()
    print('\nTraining Configuration:')
    pprint(vars(args))

    train(args)
