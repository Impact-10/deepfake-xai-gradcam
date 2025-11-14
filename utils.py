import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import timm
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -----------------------------
# DYNAMIC BATCH SIZE HELPER
# -----------------------------
def get_max_batch_size(model_name, safety_factor=0.9):
    """Estimate safe batch size for GPU memory - MAXIMIZE GPU USAGE"""
    if not torch.cuda.is_available():
        return 16  # fallback CPU batch

    device = torch.device('cuda')
    torch.cuda.empty_cache()

    # Get GPU memory info
    total_mem = torch.cuda.get_device_properties(device).total_memory // (1024 ** 2)  # MB
    
    # RTX 3050 6GB OPTIMIZED BATCH SIZES - MAXIMUM PERFORMANCE!
    if total_mem >= 10000:  # 10GB+ GPU (RTX 3080, 4080, etc.)
        aggressive_batches = {'xception': 64, 'efficientnet': 128, 'resnet50': 96}
    elif total_mem >= 8000:  # 8GB GPU (RTX 3070, 4060 Ti, etc.)
        aggressive_batches = {'xception': 48, 'efficientnet': 96, 'resnet50': 72}
    elif total_mem >= 5500:  # 6GB GPU (RTX 3050, 3060) - TUNED FOR YOUR GPU!
        aggressive_batches = {'xception': 28, 'efficientnet': 56, 'resnet50': 40}
    else:  # 4GB or less
        aggressive_batches = {'xception': 16, 'efficientnet': 32, 'resnet50': 24}
    
    return aggressive_batches.get(model_name, 32)

# -----------------------------
# DATA LOADER
# -----------------------------
class DeepfakeDataLoader:
    def __init__(self, dataset_root="dataset", batch_size=None, model_name='xception'):
        self.dataset_root = dataset_root
        # Dynamic batch size if None
        if batch_size is None:
            self.batch_size = get_max_batch_size(model_name)
        else:
            self.batch_size = batch_size
        print(f"[INFO] Using batch size: {self.batch_size}")

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # ENHANCED AUGMENTATION FOR DEEPFAKE DETECTION
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.RandomGrayscale(p=0.1),  # Help focus on structure, not color
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Simulate compression artifacts
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # Simulate occlusion
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def get_dataloaders(self):
        train_dataset = datasets.ImageFolder(
            root=os.path.join(self.dataset_root, 'train'),
            transform=self.train_transform
        )
        val_dataset = datasets.ImageFolder(
            root=os.path.join(self.dataset_root, 'val'),
            transform=self.val_test_transform
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(self.dataset_root, 'test'),
            transform=self.val_test_transform
        )

        # WINDOWS-COMPATIBLE DATA LOADING (fix multiprocessing issues)
        import platform
        if platform.system() == 'Windows':
            num_workers = 0  # Disable multiprocessing on Windows
            persistent_workers = False
            print("[INFO] Windows detected - using single-threaded data loading")
        else:
            num_workers = min(4, os.cpu_count() or 4)
            persistent_workers = True
            
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)

        return train_loader, val_loader, test_loader

# -----------------------------
# MODELS
# -----------------------------
class XceptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('xception', pretrained=True)
        # Better classifier head for deepfake detection
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.backbone(x)

class EfficientNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        # Better classifier head for deepfake detection
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.backbone(x)

class ResNet50Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Better classifier head for deepfake detection
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.backbone(x)

def get_model(model_name):
    if model_name == 'xception':
        return XceptionModel()
    elif model_name == 'efficientnet':
        return EfficientNetModel()
    elif model_name == 'resnet50':
        return ResNet50Model()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# -----------------------------
# FREEZE BACKBONE
# -----------------------------
def freeze_backbone(model, model_name):
    """Progressive unfreezing for deepfake detection with large dataset"""
    print(f"[INFO] Setting up progressive fine-tuning for {model_name}...")
    
    if model_name == 'xception':
        # Unfreeze last 30% of layers + classifier for Xception
        total_layers = len(list(model.named_parameters()))
        unfreeze_from = int(total_layers * 0.7)
        for i, (name, param) in enumerate(model.named_parameters()):
            if i >= unfreeze_from or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    elif model_name == 'efficientnet':
        # Unfreeze last 40% of layers + classifier for EfficientNet
        total_layers = len(list(model.named_parameters()))
        unfreeze_from = int(total_layers * 0.6)
        for i, (name, param) in enumerate(model.named_parameters()):
            if i >= unfreeze_from or 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    elif model_name == 'resnet50':
        # Unfreeze last 2 residual blocks + classifier for ResNet50
        for name, param in model.named_parameters():
            if 'layer4' in name or 'layer3' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

# -----------------------------
# METRICS
# -----------------------------
def calculate_metrics(y_true, y_pred, y_prob=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    return accuracy, precision, recall, f1, auc

# -----------------------------
# CHECKPOINT
# -----------------------------
def save_checkpoint(model, optimizer, epoch, val_acc, model_name, is_best=False):
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }
    if is_best:
        torch.save(checkpoint, f'checkpoints/{model_name}_best.pth')
        print(f"[INFO] Best model saved with val_acc: {val_acc:.4f}")

# -----------------------------
# VALIDATION
# -----------------------------
def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = probs > 0.5

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy().flatten())

    val_loss /= len(val_loader)
    accuracy, precision, recall, f1, auc = calculate_metrics(all_labels, all_preds, all_probs)
    return val_loss, accuracy, precision, recall, f1, auc
