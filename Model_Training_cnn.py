import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import sys
import copy
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#_________________
# pip install torch torchvision numpy matplotlib
#__________________

# --- CONFIGURATION ---
MODEL_SAVE_PATH = 'cat_classifier_custom_cnn_pytorch.pth'
NUM_EPOCHS = 10 
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PRINT_EVERY_N_BATCHES = 1

# --- CATEGORIES ---
TARGET_CATEGORIES = ['AI_Cat', 'Plushie_Cat', 'Real_Cat']
NUM_CLASSES = len(TARGET_CATEGORIES)
# ----------------------

# --- CUSTOM CNN MODEL DEFINITION ---
class SimpleCatCNN(nn.Module):
    """
    A simple, custom Convolutional Neural Network (CNN) architecture.
    Inputs are 3-channel (RGB) images, 224x224 pixels.
    """
    def __init__(self, num_classes):
        super(SimpleCatCNN, self).__init__()
        
        # --- Feature Extractor Layers ---
        self.features = nn.Sequential(
            # Conv Layer 1: Input (3x224x224) -> Output (16x112x112)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Layer 2: Input (16x112x112) -> Output (32x56x56)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Layer 3: Input (32x56x56) -> Output (64x28x28)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        )
        
        # --- Classifier Layers ---
        # Calculates the size of the flattened feature vector: 64 channels * 28 * 28 = 50176
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # Regularization
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) 
        x = self.classifier(x)
        return x

def initialize_model(num_classes: int, device: torch.device):
    """Initializes SimpleCatCNN model."""
    print("Initializing SimpleCatCNN model...")
    model = SimpleCatCNN(num_classes=num_classes)
    model = model.to(device)
    return model
# -----------------------------------


class FilteredImageFolder(Dataset):
    """
    A wrapper around ImageFolder to filter out specific classes and 
    remap their labels to a continuous range.
    """
    def __init__(self, root: str, target_classes: List[str], transform=None):
        temp_image_folder = datasets.ImageFolder(root)
        
        class_to_idx_original = {name: i for i, name in enumerate(temp_image_folder.classes)}
        target_indices_original = [
            class_to_idx_original[name] 
            for name in target_classes 
            if name in class_to_idx_original
        ]
        
        if len(target_indices_original) != len(target_classes):
             missing_classes = set(target_classes) - set(temp_image_folder.classes)
             if missing_classes:
                 print(f"Warning: Missing sub-directories for classes: {missing_classes} in {root}. Check your folder structure.")

        original_to_new_index = {
            original_idx: new_idx 
            for new_idx, original_idx in enumerate(target_indices_original)
        }
        
        self.samples = [
            (path, original_to_new_index[label]) 
            for path, label in temp_image_folder.samples
            if label in target_indices_original
        ]
        
        self.targets = [s[1] for s in self.samples]
        self.classes = target_classes
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        return image, target

def load_data(data_dir: str, target_categories: List[str], device: torch.device) -> tuple[Dict[str, DataLoader], Dict[str, int], torch.Tensor]:
    """Loads and preprocesses image data using PyTorch DataLoaders."""
    print(f"Loading data from: {data_dir}/train and {data_dir}/val")
    
    # Data preprocessing for a custom CNN (Simpler: just rescale/normalize, and augment)
    # NOTE: The Normalize values are for a *randomly initialized* CNN. 
    # For a custom CNN, mean/std is usually used or just a simple [0.5, 0.5, 0.5] if not using ImageNet weights.
    # use simple mean/std normalization here.
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # Converts to [0, 1] range
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Custom normalization
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Custom normalization
        ]),
    }

    image_datasets = {
        x: FilteredImageFolder(
            os.path.join(data_dir, x),
            target_classes=target_categories,
            transform=data_transforms[x]
        )
        for x in ['train', 'val']
    }
    
    # --- Class Imbalance Handling: Calculate Class Weights for 'train' set ---
    train_targets = image_datasets['train'].targets
    if not train_targets:
        print("FATAL ERROR: Training dataset is empty. Check data folder contents.")
        sys.exit(1)
        
    class_counts = torch.bincount(torch.tensor(train_targets), minlength=NUM_CLASSES)
    total_samples = len(train_targets)
    
    # Uses Inverse frequency weighting
    class_weights = torch.zeros(NUM_CLASSES).float()
    for i in range(NUM_CLASSES):
        count = class_counts[i].item()
        if count > 0:
            class_weights[i] = total_samples / (NUM_CLASSES * count)
        else:
            print(f"Warning: Class '{target_categories[i]}' has zero samples in the training set.")
            class_weights[i] = 0.0

    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = class_weights.to(device)
    
    print("\n--- Class Weights (Training Set) ---")
    print(f"Total Training Samples: {total_samples}")
    for i, category in enumerate(target_categories):
        count = class_counts[i].item()
        weight = class_weights[i].item()
        print(f"Class '{category}' (Idx {i}): Count={count}, Weight={weight:.4f}")
    print("----------------------------------------")
    
    class_indices = image_datasets['train'].class_to_idx
    np.save('class_indices.npy', class_indices)
    print(f"Class indices saved: {class_indices}")

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=4)
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes, class_weights

def train_model(model: nn.Module, dataloaders: Dict[str, DataLoader], 
                dataset_sizes: Dict[str, int], criterion: nn.Module, 
                optimizer: optim.Optimizer, num_epochs: int, device: torch.device):
    """Trains the PyTorch model and tracks history."""
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\n--- Starting Training ---")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            num_batches = len(dataloaders[phase])

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train' and (batch_idx + 1) % PRINT_EVERY_N_BATCHES == 0:
                    print(f"    Batch [{batch_idx + 1}/{num_batches}] - Loss: {loss.item():.4f}")

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print() 

    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    
    # Plotting the history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Validation Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.suptitle('Training History (Simple Custom CNN)', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_save_path = 'training_history_cnn.png'
    plt.savefig(plot_save_path)
    print(f"\nTraining history plotted and saved to {plot_save_path}")
    
    return model


# --- Main Execution ---

if __name__ == '__main__':
    
    # Defines data directory (fixed path to data folder)
    DATA_DIR = './data' 
    
    # Determines the device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Basic check for data existence and structure
    # NOTE: Ensure your validation folder is named 'val' or change the list below
    if not all(os.path.isdir(os.path.join(DATA_DIR, x)) for x in ['train', 'val']):
        data_path_abs = os.path.abspath(DATA_DIR)
        print(f"FATAL ERROR: Please ensure your data directory '{data_path_abs}' exists and contains 'train' and 'val' subdirectories.")
        print("If your validation folder is named 'validation', you MUST rename it to 'val' to fix the previous issue.")
        sys.exit(1)
        
    # 1. Load Data
    dataloaders, dataset_sizes, class_weights = load_data(DATA_DIR, TARGET_CATEGORIES, device)

    # 2. Initialize Model (CNN)
    model = initialize_model(NUM_CLASSES, device)

    # 3. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights) 
    
    # Optimize ALL parameters in the custom CNN
    optimizer_ft = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Train Model
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer_ft, NUM_EPOCHS, device)

    # 5. Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel weights saved to {MODEL_SAVE_PATH}")

    print("Training and plotting process finished successfully.")