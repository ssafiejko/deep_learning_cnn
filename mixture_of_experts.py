import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from base_cnn import Cinic10CNN_dropout
from pretrained_cnn import get_densenet_four_classes, get_efficientnet

class AdjustedExpert(nn.Module):
    def __init__(self, expert, output_dim, expert_output_dim):
        super(AdjustedExpert, self).__init__()
        self.expert = expert
        self.adjust_layer = nn.Linear(expert_output_dim, output_dim)

    def forward(self, x):
        x = self.expert(x)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.adjust_layer(x)
    
class MixtureOfExperts(nn.Module):
    def __init__(self, feature_dim=1280, hidden_dim=64, num_classes=10):
        super(MixtureOfExperts, self).__init__()
        
        # Ładowanie ekspertów
        base_expert = Cinic10CNN_dropout()
        base_expert.load_state_dict(torch.load('base_perf_1.pth',map_location='cpu'))
        base_expert.classifier = nn.Identity()
        
        animal_expert = Cinic10CNN_dropout()
        animal_expert.load_state_dict(torch.load('basenet_animal_expert_1.pth'))
        animal_expert.classifier = nn.Identity()

        vehicles_expert = Cinic10CNN_dropout() 
        vehicles_expert.load_state_dict(torch.load('basenet_vehicles_expert_1.pth'))
        vehicles_expert.classifier = nn.Identity()
        
        experts = [animal_expert, vehicles_expert, base_expert]

        for expert in experts:
            for param in expert.parameters():
                param.requires_grad = False

        output_dim = 1280
        base_expert_output_dim = 256 * 4 * 4
        vehicles_expert_output_dim = 256 * 4 * 4
        animal_expert_output_dim = 256 * 4 * 4

        adjusted_experts = [
            AdjustedExpert(base_expert, output_dim, base_expert_output_dim),
            AdjustedExpert(vehicles_expert, output_dim, vehicles_expert_output_dim),
            AdjustedExpert(animal_expert, output_dim, animal_expert_output_dim),
        ]
        self.experts = nn.ModuleList(adjusted_experts)
        
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False

        # Złożona bramka decyzyjna
        self.gating_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, len(experts)),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        avg_features = expert_outputs.mean(dim=1)
        gating_weights = self.gating_network(avg_features)

        gating_weights = gating_weights.unsqueeze(-1)
        weighted_features = (expert_outputs * gating_weights).sum(dim=1)

        return self.classifier(weighted_features)
