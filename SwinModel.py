import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from tqdm import tqdm
import os

def oneEpoch(model, tLoad, optimizer, criterion, epoch):
    model.train()
    totLoss = 0
    correct = 0
    total = 0
    DEVICE = "cuda" 

    pbar = tqdm(tLoad, desc="Epoch {} train".format(epoch), ncols=90)

    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        totLoss += loss.item() * images.size(0)
        temp, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=loss.item())

    return totLoss / total, correct / total


def valid(model, vLoad, criterion, epoch):
    model.eval()
    totLoss = 0
    correct = 0
    total = 0
    DEVICE = "cuda" 

    pbar = tqdm(vLoad, desc="Epoch {} valid".format(epoch), ncols=90)

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            totLoss += loss.item() * images.size(0)
            temp, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=loss.item())

    return totLoss / total, correct / total


def main():
    folder = "datset"
    DEVICE = "cuda" 
    batch = 32
    epochs = 8
    LR = 3e-4
    subpro = 4
    classes = 3
    
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=classes)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("Swin model loaded")
 

    trainTran = transforms.Compose([transforms.Resize((224, 224)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(10),transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    valTran = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    trainData = datasets.ImageFolder(os.path.join(folder, "train"), trainTran)
    valData = datasets.ImageFolder(os.path.join(folder, "val"), valTran)
    tLoad = DataLoader(trainData, batch_size=batch, shuffle=True, num_workers=subpro)
    vLoad = DataLoader(valData, batch_size=batch, shuffle=False, num_workers=subpro)

    print(" datasets loaded ")

    best = 0
    for epoch in range(1, epochs + 1):
        print("Epoch: {} ".format(epoch))
       
        trainL, trainA = oneEpoch(model, tLoad, optimizer, criterion, epoch)
        valL, valA = valid(model, vLoad, criterion, epoch)

        scheduler.step()

        
        print("Train Loss: {0} Train Acc: {1}".format(trainL, trainA))
        print("Val Loss: {0} Val Acc: {1}".format(valL, valA))

   
        if valA >= best:
            best = valA
            torch.save(model.state_dict(), "swin_model.pth")
            print(" model saved")

    print("done")
   

if __name__ == "__main__":
    main()
