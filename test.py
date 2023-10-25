import torch
import torchvision.transforms as transforms
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import os

# ReDefine CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 37 * 37, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Class Mapping
index_to_class = {
    0: "Cherry",
    1: "Strawberry",
    2: "Tomato",
}

# Define testing function
def test_model():
    # Load the saved model
    model = CNNModel(num_classes=3)
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

    total_images = len(test_image_paths)
    class_counts = {class_name: 0 for class_name in index_to_class.values()}

    for image_path in test_image_paths:
        image = Image.open(image_path)
        image = test_transform(image)
        image = Variable(image.unsqueeze(0))  # Add batch dimension
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = index_to_class[predicted.item()]

        class_counts[predicted_class] += 1

    # Report performance statistics
    print("Total number of images analyzed:", total_images)
    for class_name, count in class_counts.items():
        percentage = (count / total_images) * 100
        print(f"Number of {class_name} images: {count} ({percentage:.2f}%)")

if __name__ == '__main__':
    test_model()

