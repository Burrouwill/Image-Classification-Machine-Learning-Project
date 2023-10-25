import torch
import torchvision.transforms as transforms
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import os

# Define CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 37 * 37, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Applied dropout
        x = self.fc2(x)
        return x

# Define testing function
def test_model():
    # Load the saved model
    model = CNNModel(num_classes=NUM_CLASSES)  # Make sure NUM_CLASSES is defined
    model.load_state_dict(torch.load("model.pth"))
    model.eval()  # Set the model to evaluation mode

    # Define data preprocessing for the test data
    test_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Specify the root directory of your test data
    test_dataset_root = 'testdata/'

    # Create a custom dataset for the test data
    test_image_paths = [os.path.join(test_dataset_root, image_name) for image_name in os.listdir(test_dataset_root)]
    predictions = []

    for image_path in test_image_paths:
        image = Image.open(image_path)
        image = test_transform(image)
        image = Variable(image.unsqueeze(0))  # Add batch dimension
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        predictions.append(predicted.item())

    # Print the predictions
    for i, prediction in enumerate(predictions, 1):
        print(f"Image {i}: Predicted class index - {prediction}")

if __name__ == '__main__':
    NUM_CLASSES = 3
    test_model()


