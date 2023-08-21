
import torch
import os
from torch import nn, flatten
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.nn import ReLU
import matplotlib.pyplot as plt

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

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

#resizes images to 32x32 since images are different sizes
transform = transforms.Compose([
    transforms.Resize(desired_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#randomize changes in transformation to increase accuracy
augementation = transforms.Compose([
    transforms.Resize(desired_size),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Download training data from open datasets.
#42 total classes, multiple images with different resolutions for each
training_data = datasets.GTSRB(
    root="data",
    split="train",
    download=True,
    transform=augementation,
)

test_data = datasets.GTSRB(
    root="data",
    split="test",
    download=True,
    transform=transform,
)

BATCH_SIZE = 64

figure = plt.figure(figsize=(20, 20))
cols, rows = 17, 17

# for image_path, class_label in training_data._samples:
#     # Extract the class label from the folder name in the image_path
    
#     print(f"Image: {image_path}, Class Label: {class_label}")

for i in range(1, 222):
    
    img = training_data[i*120][0][1] #gets image
    label = training_data.__getitem__(i*120)[1] #gets labek
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(training_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)

# Define nn
#Will be CNN
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

numClasses = 43

model = NeuralNetwork(numClasses).to(device)



loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
