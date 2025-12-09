import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# 1. SHARED CONFIGURATION
# Global variables for all files
DATA_PATH = './data'
MODEL_SAVE_PATH = 'ResNet_Model.pth'
# Number of images used to train at a time
BATCH_SIZE = 32
IMG_SIZE = 224
LEARNING_RATE = 0.001
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. DATA SETUP
def get_data_loaders():
    # Reads images from folders and prepares them for the AI.
    #print("Data setup starting")
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
    except FileNotFoundError:
        print(f"Error: Could not find data at {DATA_PATH}")
        return None, None, None

    classes = full_dataset.classes
    # Splitting data into training/validation (80% Train, 20% Val)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, classes

# 3. MODEL BUILDER
def build_model(num_classes):
    # Downloads ResNet and modifies the head.
        
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model.to(DEVICE)

# 4. TRAINING ENGINE
def train_engine():
    train_loader, val_loader, class_names = get_data_loaders()
    if not train_loader: return

    model = build_model(len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    print(f"Starting Training on {DEVICE}")
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Quick validation check (just accuracy)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        print(f"Loss: {running_loss/len(train_loader):.4f} | Val Acc: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

# 5. PREDICT ENGINE
def predict_engine():
    img_path = input("Enter the path to your image (e.g., test.jpg): ")
    if not os.path.exists(img_path):
        print("File not found.")
        return

    # Hardcoded for prediction speed (so we don't have to load the dataset)
    class_names = ['Ai_Cat', 'Plushie_Cat', 'Real_Cat']
    
    model = build_model(len(class_names))
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("No saved model found. Please TRAIN first.")
        return

    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        _, pred_idx = torch.max(outputs, 1)
    
    print(f"\nPREDICTION: {class_names[pred_idx].upper()}")
    print(f"Confidence: {probs[pred_idx].item():.2f}%")
    plt.imshow(img)
    plt.title(f"{class_names[pred_idx]} ({probs[pred_idx].item():.1f}%)")
    plt.show()

if __name__ == "__main__":
    print("1. Train | 2. Predict")
    choice = input("Choice: ")
    if choice == '1': train_engine()

    elif choice == '2': predict_engine()
