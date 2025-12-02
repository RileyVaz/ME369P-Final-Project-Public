import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from sklearn.metrics import confusion_matrix

# IMPORT from the main file so we don't rewrite code
from ResNet_Model_Main import (
    get_data_loaders, 
    build_model, 
    DEVICE, 
    MODEL_SAVE_PATH
)

def evaluate_model():
    print(f"Starting eval on {DEVICE}")
    
    # 1. Reuse the data loader from the main file
    # We ignore the train_loader (_) because we only care about validation
    _, val_loader, class_names = get_data_loaders()
    
    # 2. Reuse the model builder from the main file
    model = build_model(len(class_names))
    
    # Load weights
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Could not find {MODEL_SAVE_PATH}. Please run Training first!")
        return
    
    model.to(DEVICE)
    model.eval()

    # 3. Collect Predictions
    all_preds = []
    all_labels = []

    print("Scanning validation set")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Math & Graphing
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy*100:.1f}%)')
    plt.show()

if __name__ == "__main__":
    try:
        evaluate_model()
    except ImportError:
        print("Error: You are missing libraries for the graphs.")
        print("Please run: pip install seaborn scikit-learn")