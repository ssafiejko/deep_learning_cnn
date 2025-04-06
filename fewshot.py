import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import random
from utilities import load_data
from trainer import Trainer
from base_cnn import Cinic10CNN
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.cluster import KMeans

def create_balanced_subset(dataset, fraction=0.1):
    """
    Create a subset containing a fraction of the data for each class,
    ensuring an even class distribution.
    """
    random.seed(42)
    indices_per_class = {}
    for idx, label in enumerate(dataset.targets):
        if label not in indices_per_class:
            indices_per_class[label] = []
        indices_per_class[label].append(idx)
    
    # sample a fraction of indices (at least one sample per class)
    subset_indices = []
    for label, indices in indices_per_class.items():
        n_subset = max(1, int(len(indices) * fraction))
        subset_indices.extend(random.sample(indices, n_subset))
    
    return subset_indices

# ---------------------------
# Efficient Convolutional Block
# ---------------------------
class EfficientConvBlock(nn.Module):
    """
    Uses depthwise separable convolutions for efficiency.
    """
    def __init__(self, in_channels, out_channels):
        super(EfficientConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

# ---------------------------
# Patch Embedding for Transformer Branch
# ---------------------------
class PatchEmbed(nn.Module):
    """
    Converts image into a sequence of patch embeddings using a convolution.
    For a 32x32 image with patch_size=4, we get (8x8)=64 patches.
    """
    def __init__(self, in_channels=3, embed_dim=64, patch_size=4):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

# ---------------------------
# Transformer Block
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=2):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x shape: (B, seq_len, embed_dim) -> MultiheadAttention requires (seq_len, B, embed_dim)
        x = x.transpose(0, 1)
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        fc_output = self.fc(x)
        x = x + fc_output
        x = self.norm2(x)
        return x.transpose(0, 1)

# ---------------------------
# Fusion Layer
# ---------------------------
class FusionLayer(nn.Module):
    def __init__(self, cnn_dim, trans_dim, fused_dim=64):
        super(FusionLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(cnn_dim + trans_dim, 128),
            nn.ReLU(),
            nn.Linear(128, fused_dim),        
            nn.BatchNorm1d(fused_dim)
        )
    def forward(self, cnn_features, trans_features):
        fused = torch.cat([cnn_features, trans_features], dim=-1)
        x = self.fc(fused)                     # [B, embedding_dim]
        x = F.normalize(x, p=2, dim=1)         
        return x

# ---------------------------
# Efficient Meta-Dual Fusion Network
# ---------------------------
class EfficientMetaDualFusionNet(nn.Module):
    def __init__(self, num_classes=10, cnn_channels=32, embed_dim=64, num_heads=2, patch_size=4):
        """
        cnn_channels: number of output channels from the efficient convolution branch.
        embed_dim: dimension of transformer embeddings.
        num_classes: for conventional training head (episodic training uses n-way tasks).
        """
        super(EfficientMetaDualFusionNet, self).__init__()
        # Efficient CNN branch
        self.conv_block = EfficientConvBlock(3, cnn_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Transformer branch
        self.patch_embed = PatchEmbed(in_channels=3, embed_dim=embed_dim, patch_size=patch_size)
        self.transformer = TransformerBlock(embed_dim, num_heads)
        
        # Fusion: combine CNN features (dimension=cnn_channels) and transformer features (dimension=embed_dim)
        self.fusion = FusionLayer(cnn_dim=cnn_channels, trans_dim=embed_dim, fused_dim=embed_dim)
        
        # Classifier head, not used in conventional few-shot learning
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def extract_features(self, x):
        # CNN branch
        cnn_feat = self.conv_block(x)                      # (B, cnn_channels, H, W)
        cnn_feat = self.pool(cnn_feat).view(x.size(0), -1)   # (B, cnn_channels)
        
        # Transformer branch: patch embedding
        x_patches = self.patch_embed(x)                    # (B, num_patches, embed_dim)
        trans_feat = self.transformer(x_patches)             # (B, num_patches, embed_dim)
        trans_feat = trans_feat.mean(dim=1)                # (B, embed_dim)
        
        # Fusion
        fused_feat = self.fusion(cnn_feat, trans_feat)       # (B, embed_dim)
        return fused_feat
    
    def forward(self, x):
        fused_feat = self.extract_features(x)
        return self.classifier(fused_feat)

# ---------------------------
# Episodic Dataset for Few-Shot Learning
# ---------------------------
class EpisodicDataset(Dataset):
    def __init__(self, dataset, redaug_dataset, n_way, k_shot, q_query):
        """
        dataset: a torchvision dataset that returns (image, label)
        n_way: number of classes per episode
        k_shot: number of support examples per class
        q_query: number of query examples per class
        """
        self.dataset = dataset
        self.redaug_dataset = redaug_dataset # for supports
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.label_to_indices = {}
        for idx, (img, label) in enumerate(dataset):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        self.classes = list(self.label_to_indices.keys())
    
    def __len__(self):
        # just leave it at that
        return 10000
    
    def __getitem__(self, index):
        sampled_classes = random.sample(self.classes, self.n_way)
        support_imgs = []
        support_labels = []
        query_imgs = []
        query_labels = []
        for i, cls in enumerate(sampled_classes):
            indices = random.sample(self.label_to_indices[cls], self.k_shot + self.q_query)
            support_idx = indices[:self.k_shot]
            query_idx = indices[self.k_shot:]
            for idx in support_idx:
                img, _ = self.redaug_dataset[idx]
                support_imgs.append(img)
                support_labels.append(i)  # relabel classes 
            for idx in query_idx:
                img, _ = self.dataset[idx]
                query_imgs.append(img)
                query_labels.append(i)
        support_imgs = torch.stack(support_imgs)  # (n_way*k_shot, C, H, W)
        support_labels = torch.tensor(support_labels)
        query_imgs = torch.stack(query_imgs)      # (n_way*q_query, C, H, W)
        query_labels = torch.tensor(query_labels)
        return support_imgs, support_labels, query_imgs, query_labels
    

# ---------------------------
# Utility Functions for Episodic Training
# ---------------------------
def compute_prototypes(features, labels, n_way, k_shot):
    # features: (n_way*k_shot, feature_dim)
    # labels: (n_way*k_shot)
    prototypes = []
    for i in range(n_way):
        class_features = features[labels == i]
        prototype = class_features.mean(dim=0)
        prototypes.append(prototype)
    prototypes = torch.stack(prototypes)  # (n_way, feature_dim)
    return prototypes

def compute_logits(query_features, prototypes):
    dists = torch.cdist(query_features, prototypes, p=2)  # (n_query, n_way)
    logits = -(dists ** 2)
    return logits

def compute_mean_std(dataset, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        # shape of images: (B, C, H, W)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # (B, C, H*W)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std
# ---------------------------
# Training Loop
# ---------------------------
def train_few_shot(model, dataloader, optimizer, criterion, device, n_way, k_shot, q_query, num_episodes):
    model.train()
    training_losses = []
    running_loss = 0.0
    for episode in range(num_episodes):
        support_imgs, support_labels, query_imgs, query_labels = next(iter(dataloader))
        support_imgs = support_imgs.squeeze(0).to(device)
        support_labels = support_labels.squeeze(0).to(device)
        query_imgs = query_imgs.squeeze(0).to(device)
        query_labels = query_labels.squeeze(0).to(device)
        
        support_features = model.extract_features(support_imgs)
        query_features = model.extract_features(query_imgs)
        
        prototypes = compute_prototypes(support_features, support_labels, n_way, k_shot)
        logits = compute_logits(query_features, prototypes)
        
        loss = criterion(logits, query_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (episode+1) % 100 == 0:
            avg_loss = running_loss / 100
            print(f"Episode [{episode+1}/{num_episodes}], Loss: {avg_loss:.4f}")
            training_losses.append(avg_loss)
            running_loss = 0.0
    return training_losses

def _get_augmented_transform(mean, std):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.AutoAugment(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        ])
    return transform

def _get_base_transform(mean=None, std=None, normalized=False):
    base_aug = [transforms.Resize((32, 32)), transforms.ToTensor()]
    if normalized:
        base_aug.append(transforms.Normalize(mean=mean, std=std))
    transform = transforms.Compose(base_aug)
    return transform

def predict_with_subprototypes(model, sub_prototypes, test_image, device='cpu'):
    """
    Compare the test_image embedding to each sub-prototype in every class,
    pick the class with the closest sub-prototype.
    
    sub_prototypes: { class_idx: tensor of shape [n_sub, embedding_dim], ... }
    test_image: a single image (tensor or PIL)
    """
    model.eval()
    
    # 1) Embed the test image
    if isinstance(test_image, torch.Tensor):
        x = test_image.unsqueeze(0).to(device)
    else:
        x = transforms.ToTensor()(test_image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.extract_features(x)  # [1, embedding_dim]
    
    emb = emb.squeeze(0)  # shape [embedding_dim]
    
    best_dist = float('inf')
    best_class = None

    for class_idx, proto_mat in sub_prototypes.items():
        if len(proto_mat) == 0:
            continue
        
        proto_mat = proto_mat.to(device)

        # compute distances to all sub-prototypes in this class
        diff = proto_mat - emb.unsqueeze(0)   
        dists = (diff ** 2).sum(dim=1)       

        class_min_dist = dists.min().item()
        if class_min_dist < best_dist:
            best_dist = class_min_dist
            best_class = class_idx
    
    return best_class

def build_sub_prototypes(class_to_embeddings, n_sub=2):
    """
    For each class, run k-means to find 'n_sub' clusters in its embeddings.
    Return a dict: class_idx -> list of sub-prototype tensors.
    """
    sub_prototypes = {}
    for class_idx, emb_list in class_to_embeddings.items():
        if len(emb_list) == 0:
            sub_prototypes[class_idx] = []
            continue
        
        emb_tensor = torch.cat(emb_list, dim=0).numpy() # into (num_embeddings, embedding_dim)
        
        # If embeddings < n_sub, might reduce n_sub temporarily to avoid errors
        n_clusters = min(n_sub, emb_tensor.shape[0])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(emb_tensor)
        centers = kmeans.cluster_centers_
        
        # Convert back to torch
        centers_torch = torch.tensor(centers, dtype=torch.float32)
        sub_prototypes[class_idx] = centers_torch  # (n_sub, embedding_dim)
    return sub_prototypes

def collect_class_embeddings(model, dataset, device='cpu'):
    """
    Return a dictionary mapping class_idx -> list of embeddings 
    from the trained feature extractor.
    """
    model.eval()
    
    class_to_embeddings = {}
    num_classes = 10  # for CINIC-10
    for c in range(num_classes):
        class_to_embeddings[c] = []

    with torch.no_grad():
        for i in range(len(dataset)):
            img, label = dataset[i]
            if isinstance(img, torch.Tensor):
                x = img.unsqueeze(0).to(device)  # [1, C, H, W]
            else:
                # If dataset returns PIL images, convert to tensor
                x = transforms.ToTensor()(img).unsqueeze(0).to(device)

            emb = model.extract_features(x)  # shape [1, embedding_dim]
            class_to_embeddings[label].append(emb.cpu())

    return class_to_embeddings

def predict_single_image(model, prototypes, test_image, device='cpu'):
    """
    Predict the class of a single test image using a trained ProtoNet and
    class prototypes from build_class_prototypes().
    
    prototypes: dict of {class_idx: prototype_vector}
    test_image: a single image, either a Tensor [3, H, W] or a PIL image
    """
    model.eval()
    
    if isinstance(test_image, torch.Tensor):
        x = test_image.unsqueeze(0).to(device)  # [1, 3, H, W]
    else:
        # If it's a PIL image, convert to tensor with standard transforms
        x = transforms.ToTensor()(test_image).unsqueeze(0).to(device)
        
    with torch.no_grad():
        test_emb = model.extract_features(x)  # shape [1, embedding_dim]
    
    # distances to each class prototype
    distances = []
    class_indices = []
    for class_idx, proto in prototypes.items():
        proto = proto.to(device)           # shape [embedding_dim]
        diff = test_emb - proto.unsqueeze(0)  # shape [1, embedding_dim]
        dist = (diff ** 2).sum().item()    # scalar
        distances.append(dist)
        class_indices.append(class_idx)
    
    # Pick class with minimum distance
    distances = torch.tensor(distances)
    min_dist_idx = torch.argmin(distances).item()
    predicted_class = class_indices[min_dist_idx]
    return predicted_class

def build_class_prototypes(class_to_embeddings):
    """
    Using a trained ProtoNet model, compute a prototype for each class
    by averaging the embeddings of all images belonging to that class.
    This approach uses the entire dataset for demonstration.
    """
    prototypes = {}
    for class_idx in range(10):
        if len(class_to_embeddings[class_idx]) == 0:
            # For CINIC-10, each class should be present.
            continue
        
        all_embs = torch.cat(class_to_embeddings[class_idx], dim=0)  # shape [num_imgs_for_class, embedding_dim]
        class_proto = all_embs.mean(dim=0)                           # shape [embedding_dim]
        prototypes[class_idx] = class_proto
    
    return prototypes