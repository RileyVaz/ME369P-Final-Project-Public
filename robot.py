import torch
from torchvision import transforms
from PIL import Image
import timm
import pybullet as p
import pybullet_data
import time
import math


def image(model_path, img_path):
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


def main(image_path, model_path):
    
    idx, confidence = image(model_path,image_path)
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
    modelpath = input("Please enter path to model: ") 
    while True: 
        imagepath = input("Please enter path to image: ")
        main(imagepath, modelpath)
        if input("test more images(yes or no)? ") in "no":
            
            break
        
        elif input("test different model(yes or no)? ") in  "yes":
            modelpath = input("Please enter path to model: ") 
          

            
        # "C:\Users\riley\fake.jpg"
        #"C:\Users\riley\best_swin_model.pth"



