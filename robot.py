import torch
from torchvision import transforms,models
from PIL import Image
import timm
import pybullet as p
import pybullet_data
import time
import math


def image_swain(model_path, img_path):
    device = torch.device("cuda")
    model = timm.create_model("swin_tiny_patch4_window7_224",pretrained=True,num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    cat = Image.open(img_path).convert("RGB")
    tensor = transform(cat).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        conf, idx = torch.max( torch.softmax(output, dim=1), dim=1)

    return idx.item(), conf.item()

def image_resnet(model_path, img_path):
    device = torch.device("cuda")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    cat = Image.open(img_path).convert("RGB")
    tensor = transform(cat).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, idx = torch.max(probs, dim=1)

    return idx.item(), conf.item()

class SimpleCatCNN(torch.nn.Module):
    """
    A simple, custom Convolutional Neural Network (CNN) architecture.
    Inputs are 3-channel (RGB) images, 224x224 pixels.
    """
    def __init__(self, num_classes):
        super(SimpleCatCNN, self).__init__()
        
        # --- Feature Extractor Layers ---
        self.features = torch.nn.Sequential(
            # Conv Layer 1: Input (3x224x224) -> Output (16x112x112)
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Layer 2: Input (16x112x112) -> Output (32x56x56)
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Layer 3: Input (32x56x56) -> Output (64x28x28)
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
        )
        
        # --- Classifier Layers ---
        # Calculates the size of the flattened feature vector: 64 channels * 28 * 28 = 50176
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5), # Regularization
            torch.nn.Linear(64 * 28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) 
        x = self.classifier(x)
        return x


def image_cnn(model_path, img_path):
    device = torch.device("cuda")
    model = SimpleCatCNN(3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    cat = Image.open(img_path).convert("RGB")
    tensor = transform(cat).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, idx = torch.max(probs, 1)

    return idx.item(), conf.item()


def move(robot_id, target):

    ori = p.getQuaternionFromEuler([math.pi, 0, 0])
    target_joints = p.calculateInverseKinematics(robot_id, 6, target, ori)

    current_joints = [p.getJointState(robot_id, j)[0] for j in range(7)]

    # Slowly interpolate
    for i in range(500):
 
        interpolated = [current_joints[j] * (1 - i / 500) + target_joints[j] * i / 500 for j in range(7)]
        for k in range(7):p.setJointMotorControl2(robot_id, k, p.POSITION_CONTROL, targetPosition=interpolated[k], force=300)

        p.stepSimulation()
        time.sleep(.005)


def scale():
  
    x = 0.0
    y = 0.0
    interval = 0.4
    
    p.addUserDebugLine([x, y, 0],[x, y,interval], [0, 0, 0], lineWidth=3, lifeTime=0)
    
    for i in range(5):
        percent = i / 4
        z = (percent * interval)
        color = [1 - percent, percent, 0]
        
        p.addUserDebugLine( [x+1, y+1, z], [x-1, y - 1, z], color, lineWidth=3, lifeTime=0)
        
        p.addUserDebugText( "{}%".format(int(percent*100)), [x, y - 0.25, z], textColorRGB=[0, 0, 0], textSize=1.5, lifeTime=0)
  
    p.addUserDebugText("CONFIDENCE", [x, y - 0.2, interval + 0.1], [0,0,0], textSize=1.2)

def zoneD(zones):
    
    size = 0.2
    color = [1, 1, 1, 1]  
    for name, (x, y) in zones.items():

        p.addUserDebugLine([x-size, y-size, 0], [x+size, y-size, 0], color, 2)

        p.addUserDebugLine([x+size, y-size, 0], [x+size, y+size, 0], color, 2)

        p.addUserDebugLine([x+size, y+size, 0],[x-size, y+size, 0], color, 2)

        p.addUserDebugLine([x-size, y+size, 0], [x-size, y-size, 0], color, 2)

        p.addUserDebugText(text=name, textPosition=[x, y, 0.05], textColorRGB=[0, 0, 0], textSize=1.2, lifeTime=0)


def main(image_path, model_path,model):
    if "swain" in model:
        idx, confidence = image_swain(model_path,image_path)
    elif "resnet" in model:
        idx, confidence =  image_resnet(model_path,image_path)
    else: 
        idx, confidence =  image_cnn(model_path,image_path)


    CATagories = ["Ai",  "Plushie","real"]
    cat = CATagories[idx]

    print("CATegory:", cat)
    print("Confidence: {} %".format( confidence*100))

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.resetDebugVisualizerCamera(1.6, 90, -30, [0.2, 0.2, 0.5])
    p.loadURDF("plane.urdf")
    
    start = [0, 0, 0]
    robot = p.loadURDF("kuka_iiwa/model.urdf", start, p.getQuaternionFromEuler(start))


    zones = {"Ai": [-0.5, -0.5], "Plushie":  [0.5, 0.5],"real": [0.5, -0.5]}
    zoneD(zones)
    zone = zones[cat]
    target = [zone[0], zone[1], 0.4*confidence]
    scale()
    time.sleep(1)
    

    print("Moving robot to zone for class {}:".format(cat), target)
    move(robot, target) 
    time.sleep(5)
    p.disconnect()
    print("Done")


if __name__ == "__main__":
    model = input("Please enter type of model(swain, resnet, or cnn): ") 
    modelpath = input("Please enter path to model: ") 
    while True: 
        imagepath = input("Please enter path to image: ")
        main(imagepath, modelpath,model)
        if input("test more images(yes or no)? ") in "no":
            
            break
        
        elif input("test different model(yes or no)? ") in  "yes":
            model = input("Please enter type of model(swain, resnet, or cnn): ")
            modelpath = input("Please enter path to model: ") 
          

            
        # "C:\Users\riley\fake.jpg"
        #"C:\Users\riley\best_swin_model.pth"



