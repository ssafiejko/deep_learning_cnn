from trainer import Trainer
from utilities import load_augmented_data, load_data, set_seed
from pretrained_cnn import get_densenet, get_efficientnet, get_resnet18
from fewshot import create_balanced_subset, _get_base_transform, _get_augmented_transform, compute_mean_std
from base_cnn import Cinic10CNN
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import warnings
from pprint import pprint

TRAIN_DIRECTORY = './train'
VALID_DIRECTORY = './valid'
TEST_DIRECTORY = './test'

CRITERION = nn.CrossEntropyLoss() # Other loss functions were not analyzed

FRAC_BATCHSIZE = {
    0.001: 4,
    0.002: 4,
    0.005: 8,
    0.01: 16, 
    0.02: 32, 
    0.05: 64, 
    0.1: 64 
}

def train(args):
    model = _get_model(args.architecture, args.model_name, args.dropout)
    optimizer = _get_optimizer(model.parameters(), args.lr, args.optimizer, weight_decay=args.weight_decay)
    train_loader, valid_loader, test_loader = _get_dataloaders(args.augmentations, args.batch_size, args.double_augment, args.fraction)

    trainer = Trainer(model, optimizer, CRITERION, train_loader, valid_loader, test_loader)

    set_seed()
    trainer.train_multiple(n=args.num_trainings, n_epochs=args.num_epochs)

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

def _get_model(model_architecture: str, model_name: str, dropout: float):
    """
    Instantiate and return a model based on the given architecture.

    Parameters:
    -----------
    model_architecture : str
        The architecture type of the model. Options: 'custom', 'resnet', 'densenet', 'efficientnet'.
    model_name : str
        The name to assign to the model instance.
    dropout : float
        Dropout rate for the 'custom' model architecture. Ignored for other architectures.

    Returns:
    --------
    model : nn.Module
        An instance of the selected model architecture with the specified configuration.
    """
    architectures = {
        'custom': Cinic10CNN,
        'resnet': get_resnet18,
        'densenet': get_densenet,
        'efficientnet': get_efficientnet
    }

    if model_architecture not in architectures:
        raise ValueError(f"Unsupported model architecture: {model_architecture}")

    if model_architecture != 'custom' and dropout != 0:
        warnings.warn("Non-custom architecture selected with non-zero dropout, ignoring dropout parameter.",
                      UserWarning)

    if model_architecture == 'custom':
        model = architectures[model_architecture](dropout=dropout)
    else:
        model = architectures[model_architecture]()

    model.name = model_name
    return model

def _get_dataloaders(augmentations, batch_size, double_augment, fraction):
    if double_augment == 0:
        train_loader, valid_loader, test_loader = load_data(TRAIN_DIRECTORY, VALID_DIRECTORY, TEST_DIRECTORY, augmentations, batch_size)
    else:
        train_loader, valid_loader, test_loader = load_augmented_data(TRAIN_DIRECTORY, VALID_DIRECTORY, TEST_DIRECTORY, augmentations, batch_size)
    
    if fraction != 1:
        base_transform =  _get_base_transform()
        dataset_tmp = datasets.ImageFolder(TRAIN_DIRECTORY, transform=base_transform) # No transforms, used for mean and std calculations 
        dataset_indices = create_balanced_subset(dataset_tmp, fraction=fraction)
        mean, std = compute_mean_std(Subset(dataset_tmp, dataset_indices))

        transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.AutoAugment(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        ])

        full_dataset = datasets.ImageFolder(TRAIN_DIRECTORY, transform=transform_train)

        dataset = Subset(full_dataset, dataset_indices)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with configurable settings.")
    parser.add_argument('--architecture', type=str, default='custom', help='Model architecture: [custom, efficientnet, resnet, densenet]')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer name')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
    parser.add_argument('--num_trainings', type=int, default=3, help='Number of training runs')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs per run')
    parser.add_argument('--model_name', type=str, default='tmp', help='Model name identifier')
    parser.add_argument('--double_augment', type=int, default=0, help='Use double data with one augmented segment Yes/No [1/0]')
    parser.add_argument('--fraction', type=float, default=1.0, help='Fraction of the training data to be used.')
    parser.add_argument(
        '--augmentations',
        nargs='+', 
        type=str,
        default=[],
        help="List of augmentations to apply: ['random_crop', 'horizontal_flip', 'rotation', 'color_jitter', 'cutout', 'autoaugment']")

    args = parser.parse_args()

    print('\nTraining Configuration:')
    pprint(vars(args))

    train(args)