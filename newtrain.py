
# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# %%
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
print(device)


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
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 28 * 28)
        x = self.classifier(x)
        return x

# %%
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%
# Load datasets
train_dataset = torchvision.datasets.ImageFolder(root="train", transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root="test", transform=transform)


# %%
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %%
# Initialize the model
model = ChestCNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# %%
# Evaluating the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {correct / total}")

# %%
torch.save(model.state_dict(), 'phase3.pth')


# %%
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

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


# %%
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
val_dataset = torchvision.datasets.ImageFolder(root="val", transform=transform)  # Use a separate validation dataset
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Evaluate the model on the validation set
true_labels = []
predicted_labels = []

model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.to(device)  # Move the model to the same device as the inputs
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# Calculate the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# %%
# User input image path
image_path = "val/TURBERCULOSIS/Tuberculosis-652.png"

# %%
def classify_image(image_path, model, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    image = image.to(device)  # Move the input tensor to the GPU
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()


# %%
# Mapping from class index to class label
class_labels=["COVID", "Normal","Pneumonia","TB"]
# %%
# Display the prediction
print("Predicted class:", class_labels[predicted_class_index])




# %%
