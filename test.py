import torch
import os
from torch import nn, flatten
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.nn import ReLU
import matplotlib.pyplot as plt
import random
# from main import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class NeuralNetwork(nn.Module):
    def __init__(self, numClasses):
        super(NeuralNetwork, self).__init__()
        # 2 convolutional layers Convolution creates a feature map
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(5,5), stride=1) #in_channels, out_channels, kernel_size, stride\
        self.relu1 = ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5,5), stride=1) #Converts 32 channels to 64
        self.relu2 = ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        #Fully connected layer, takes inputs to the output layer


        self.fc1 = nn.Linear(4624, 800) 
        self.relu3 = ReLU()

        self.fc2 = nn.Linear(800, 150)
        self.relu4 = ReLU()

        self.fc3 = nn.Linear(150, numClasses) 
        self.logSoftmax = nn.LogSoftmax(dim=1)
        
        #
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        # Flatten the feature maps
        x = flatten(x, 1)
        
        # Apply fully connected layers and activation functions
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        x = self.relu4(x)

        x = self.fc3(x)
        x = self.logSoftmax(x)
        
        return x
    
    
labels = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing veh over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Veh > 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing veh > 3.5 tons"
}

desired_size = (80, 80)

transform = transforms.Compose([
    transforms.Resize(desired_size),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

training_data = datasets.GTSRB(
    root="data",
    split="train",
    download=False,
    transform=transform,
)

test_data = datasets.GTSRB(
    root="data",
    split="test",
    download=False,
    transform=transform,
)

test_dataloader = DataLoader(test_data,
    batch_size=1,
    shuffle=False # don't necessarily have to shuffle the testing data
)

model = NeuralNetwork(43)
model.load_state_dict(torch.load("model.pth"))
model.eval()

cols, rows = 1, 1
# img = training_data[i*120][0][1] #gets image
# label = training_data.__getitem__(i*120)[1] #gets label

#print size of test data
print(len(test_data)) #12630

# example
# x, y = test_data[0][0], test_data[0][1]
# with torch.no_grad():
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')

while(True): 
    figure = plt.figure(figsize=(7.5, 7.5))
    rand = random.randint(0, len(test_data))
    img = test_data[rand][0][1] #gets image
    label = test_data.__getitem__(rand)[1] #gets label
    figure.add_subplot()
    plt.axis("off")
    with torch.no_grad():
        # input_tensor = test_data[rand][0][1].unsqueeze(0).unsqueeze(0)
        pred = model(test_data[rand][0].unsqueeze(0))
    predicted_class = torch.argmax(pred).item()
    plt.imshow(img.squeeze(), cmap="gray")
    plt.text(0, 85, "Actual: " + labels[label], fontsize=24, ha='left')  # Subtitle 1
    plt.text(0, 90, "Predicted: " + labels[predicted_class], fontsize=24, ha='left')  # Subtitle 2
    plt.subplots_adjust(top=1) 
    plt.show()
        


# for i in range(0, 222):
    
#     img = training_data[i*120][0][1] #gets image
#     label = training_data.__getitem__(i*120)[1] #gets labek
#     figure.add_subplot(rows, cols, i)
#     plt.title(label)
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()