# ME369P-Final-Project-Public
1.The purpose of this project is to differentiate between an AI cat, a real cat, or a plushie cat. This is accomplished by training 3 models: a CNN, ResNet, and a Swin Transformer. The Swin Transformer performed the best. The models can be downloaded from the following links, and the data used to train these models. 
2. After an image is classified, the end effector of a robot is moved to a corresponding x,y coordinate based on its classification and then to a corresponding Z coordinate based on the model's confidence value. 

## model link 
https://drive.google.com/file/d/1LmHvKBR9CANGtaBeEO-KHJPdY2GyRZa_/view?usp=sharing
## data link 
https://drive.google.com/drive/folders/1-UtQ0X5nTJHMcUOAaTisVJr9Kmh7YTsy?usp=sharing
## quick guide 
The robot.py file takes in an input of a model that the user desires to use, and an image that the user wishes to evaluate.

## Running the CNN model

1. To run the CNN model, the following packages are requiredd in order to run
   pip install torch torchvision numpy matplotlib
