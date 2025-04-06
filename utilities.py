from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(train_dir, valid_dir, test_dir, augmentations, batch_size=64):
    """
    Loads and preprocesses image data from the specified directories and returns data loaders for training, validation, and testing.

    Args:
        train_dir (str): Path to the training data directory.
        valid_dir (str): Path to the validation data directory.
        test_dir (str): Path to the testing data directory.
        augmentations (list): List of image augmentations to apply to the training data. Choose from ['random_crop', 'horizontal_flip', 'rotation', 'color_jitter', 'cutout', 'autoaugment'].
        batch_size (int, optional): Number of samples per batch. Default is 64.

    Returns:
        tuple: A tuple containing three DataLoader objects for training, validation, and testing datasets.
    """
    # Data transformations
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    applied_augmentations = [transforms.Resize((32, 32))]
    
    if 'random_crop' in augmentations:
        print('Random crop added')
        applied_augmentations.append(transforms.RandomResizedCrop(32, scale=(0.8, 1.0)))
    if 'horizontal_flip' in augmentations:
        print('Horizontal flip added')
        applied_augmentations.append(transforms.RandomHorizontalFlip())
    if 'rotation' in augmentations:
        print('Rotation added')
        applied_augmentations.append(transforms.RandomRotation(15))
    if 'color_jitter' in augmentations:
        print('Color jitter added')
        applied_augmentations.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
    if 'autoaugment' in augmentations:
        print('AutoAugment added')
        applied_augmentations.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    
    applied_augmentations.append(transforms.ToTensor())
    applied_augmentations.append(transforms.Normalize(mean=cinic_mean, std=cinic_std))
    
    if 'cutout' in augmentations:
        print('Cutout added')
        applied_augmentations.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0))
    
    train_transform = transforms.Compose(applied_augmentations)

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std),
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    return train_loader, valid_loader, test_loader

def load_history(filename: str):
    """
    Loads a history object from a pickle file.
    
    Args:
        filename (str): Path to the pickle file.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def plot_best_history(filenames: list):
    """
    Plots the training and validation accuracy and loss of the best model from a list of history files.
    
    Args:
        filenames (list): List of paths to the history files.
    """
    best_val_acc = 0
    best_history = None

    for filename in filenames:
        history = load_history(filename)
        max_val_acc = np.max(history['val_acc_history'])
        
        if max_val_acc > best_val_acc:
            best_val_acc = max_val_acc
            best_history = history

    if best_history is None:
        print("No valid history found.")
        return
    print(f'Best validation accuracy: {best_val_acc}')
    train_acc_history = best_history['train_acc_history']
    val_acc_history = best_history['val_acc_history']
    train_loss_history = best_history['train_loss_history']
    val_loss_history = best_history['val_loss_history']

    epochs = range(1, len(train_acc_history) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc_history, 'b', label='Train Accuracy')
    plt.plot(epochs, val_acc_history, 'r', label='Validation Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss_history, 'b', label='Train Loss')
    plt.plot(epochs, val_loss_history, 'r', label='Validation Loss')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def show_sample_images(train_loader):
    """
    Displays 10 sample images from the training data loader.
    
    Args:
        train_loader (DataLoader): The training data loader, assumes `normalized` input.
    """
    images, labels = next(iter(train_loader))
    images = images[:10] 
    
    fig, axes = plt.subplots(1, 10, figsize=(15, 5))
    for img, ax in zip(images, axes):
        img = img.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        img = img * np.array([0.24205776, 0.23828046, 0.25874835]) + np.array([0.47889522, 0.47227842, 0.43047404])  # Denormalize
        img = np.clip(img, 0, 1)  # Clip values to valid range
        ax.imshow(img)
        ax.axis('off')
    plt.show()

import torch
from sklearn.metrics import f1_score, precision_score, recall_score

def test_model(model, model_path, test_loader):
    
    criterion = nn.CrossEntropyLoss()
    device= torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    model.load_state_dict(torch.load(model_path))

    model.eval()
    
    test_loss, test_correct = 0.0, 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    test_acc = test_correct.float() / len(test_loader.dataset)
    test_f1_score = f1_score(all_labels, all_preds, average='macro')
    test_precision = precision_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')

    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}', 
          f'Test F1: {test_f1_score:.4f}, Test Precision: {test_precision:.4f}',
          f'Test Recall: {test_recall:.4f}')
    return test_loss, test_acc, test_f1_score, test_precision, test_recall


import copy
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch

class DoubleAugmentedDataset(torch.utils.data.Dataset):
    """
    Custom dataset that returns two versions for each image:
    one with basic augmentations and one with basic + a complex augmentation.
    """
    def __init__(self, base_dataset, transform_basic, transform_extended):
        self.base_dataset = base_dataset
        self.transform_basic = transform_basic
        self.transform_extended = transform_extended

    def __len__(self):
        return 2 * len(self.base_dataset)

    def __getitem__(self, index):
        if index < len(self.base_dataset):
            img, label = self.base_dataset[index]
            return self.transform_basic(img), label
        else:
            img, label = self.base_dataset[index - len(self.base_dataset)]
            return self.transform_extended(img), label

def load_augmented_data(train_dir, valid_dir, test_dir, augmentations, batch_size=64, fraction=1):
    """
    Loads image data and applies all selected basic augmentations. For the training dataset,
    each image is duplicated: one version is transformed with the basic augmentations,
    and the second version is transformed with both basic augmentations and one complex augmentation.
    The validation and testing datasets are processed using a test transform.

    Args:
        train_dir (str): Path to the training data directory.
        valid_dir (str): Path to the validation data directory.
        test_dir (str): Path to the testing data directory.
        augmentations (list): List of augmentations. For basic augmentations, choose from 
                              ['random_crop', 'horizontal_flip', 'rotation', 'color_jitter'].
                              For complex augmentations, choose from ['cutout', 'autoaugment'].
        batch_size (int, optional): Number of samples per batch. Default is 64.

    Returns:
        tuple: DataLoader objects for training, validation, and testing datasets.
    """
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    # Build basic augmentation transforms
    basic_augmentations = [transforms.Resize((32, 32))]
    if 'random_crop' in augmentations:
        basic_augmentations.append(transforms.RandomResizedCrop(32, scale=(0.8, 1.0)))
    if 'horizontal_flip' in augmentations:
        basic_augmentations.append(transforms.RandomHorizontalFlip())
    if 'rotation' in augmentations:
        basic_augmentations.append(transforms.RandomRotation(15))
    if 'color_jitter' in augmentations:
        basic_augmentations.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
    
    # Append ToTensor and Normalize to complete the pipeline.
    basic_augmentations.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)
    ])
    train_transform_basic = transforms.Compose(basic_augmentations)

    # Choose one complex augmentation if provided
    complex_augmentations = []
    if 'cutout' in augmentations:
        complex_augmentations.append(transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0))
    if 'autoaugment' in augmentations:
        complex_augmentations.append(transforms.AutoAugment())
    chosen_complex_aug = complex_augmentations[0] if complex_augmentations else None

    # Create extended transformation: start from a copy of the basic augmentations list
    extended_augmentations = copy.deepcopy(basic_augmentations)
    if chosen_complex_aug:
        if isinstance(chosen_complex_aug, transforms.RandomErasing):
            insertion_index = -1  # before the Normalize transform
        else:
            insertion_index = -2  # before ToTensor
        extended_augmentations.insert(insertion_index, chosen_complex_aug)
    train_transform_extended = transforms.Compose(extended_augmentations)

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std),
    ])

    # Load base datasets
    base_train_dataset = datasets.ImageFolder(train_dir)
    paired_train_dataset = DoubleAugmentedDataset(base_train_dataset, train_transform_basic, train_transform_extended)
    
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    return train_loader, valid_loader, test_loader