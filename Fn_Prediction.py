import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import numpy as np

def my_filebrowser():
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    imgfile = filedialog.askopenfilename(initialdir = "test",title = "Select Image",filetypes = (("Img files","*.*"),("all files","*.*")))
    root.destroy()
    return imgfile
# %%
# Define the CNN architecture
class ChestCNN(nn.Module):
    def __init__(self):
        super(ChestCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # 4 classes: COVID, pneumonia, TB, normal
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 28 * 28)
        x = self.classifier(x)
        return x

# %%
# Load the trained model
model = ChestCNN()
model.load_state_dict(torch.load("phase3.pth", map_location=torch.device('cpu')))  # Load model weights
model.eval()  # Set the model to evaluation mode


# %%
# Define transformations for user input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%
# Function to classify the user input image
def classify_image(image_path, model, transform):
    image = Image.open(image_path).convert('RGB')  # Load image
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()  # Return the predicted class index

def makePrediction():
    image_path = my_filebrowser()

    print(image_path)
    
    # Classify the user input image
    predicted_class_index = classify_image(image_path, model, transform)
    # Mapping from class index to class label
    class_labels = ["COVID", "Normal","Pneumonia","TB",]
# Display the prediction

    pred=class_labels[predicted_class_index]
    print("Predicted class:",pred)

##severity
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    count1=0
    for val in range(124,130):
        pix1 = np.sum(img == val)
        count1=count1+pix1
    count1=count1/1000
    ####print(count1)
    stage=''
    if predicted_class_index!=1:
        if(count1<=10):
            stage='stage 1'
        elif (count1<=25):
            stage='stage 2'
        else:
            stage='stage 3'

    pred=pred+"  "+stage
    return pred




def Pred2(image_path):
    
    # Classify the user input image
    predicted_class_index = classify_image(image_path, model, transform)
    # Mapping from class index to class label
    class_labels = ["COVID", "Normal","Pneumonia","TB",]
# Display the prediction
    pred=class_labels[predicted_class_index]
    ##print("Predicted class:",pred)
    
    return predicted_class_index

    





