from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import torch.optim as optim

def get_efficientnet():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_b0(pretrained=True)

    # Replace the fully connected layer
    num_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, 10)  
    )

    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(device)

    model.__name__ = 'efficientnet_b0'
    model.name = 'efficientnet_b0'
    return model

def get_densenet():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    model = models.densenet121(pretrained=False)

    # Replace the classifier layer
    num_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, 10)  
    )

    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(device)

    model.__name__ = 'densenet121'
    model.name = 'densenet121'
    return model
def get_densenet_four_classes():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    model = models.densenet121(pretrained=False)

    # Replace the classifier layer
    num_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, 4)  
    )

    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(device)

    model.__name__ = 'densenet121'
    model.name = 'densenet121'
    return model
    
def get_resnet18():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=True)

    # Replace the fully connected layer
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, 10)  
    )

    for param in model.fc.parameters():
        param.requires_grad = True

    model.to(device)

    model.__name__ = 'resnet18'
    model.name = 'resnet18'
    return model