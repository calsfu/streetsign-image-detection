
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.nn import ReLU
import matplotlib.pyplot as plt


desired_size = (32, 32)
#resizes images to 32x32
transform = transforms.Compose([
    transforms.Resize(desired_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Download training data from open datasets.
#42 total classes, multiple images with different resolutions for each
training_data = datasets.GTSRB(
    root="data",
    split="train",
    download=True,
    transform=ToTensor(),
)

test_data = datasets.GTSRB(
    root="data",
    split="test",
    download=True,
    transform=transform,
)



BATCH_SIZE = 32

figure = plt.figure(figsize=(20, 20))
cols, rows = 16, 16

# for image_path, class_label in training_data._samples:
#     # Extract the class label from the folder name in the image_path
    
#     print(f"Image: {image_path}, Class Label: {class_label}")

for i in range(1, 217):
    
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
        self.conv1 = nn.Conv2d(1, 32, 3, 1) #in_channels, out_channels, kernel_size, stride\
        self.relu1 = ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, 1) #Converts 32 channels to 64
        self.relu2 = ReLU()
        #Fully connected layer, takes inputs to the output layer

        self.fc1 = nn.Linear(9216, 128) #9216 is the number of inputs
        self.relu3 = ReLU()

        self.fc2 = nn.Linear(9216, numClasses)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        #
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        # Flatten the feature maps
        x = x.view(x.size(0), -1)  # Flatten to 1D
        
        # Apply fully connected layers and activation functions
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        x = self.logSoftmax(x)
        
        return x

numClasses = 42

model = NeuralNetwork(numClasses).to(device)



loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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

# def test(dataloader, model):
#     size = len(dataloader.dataset)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= size
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# epochs = 15
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model)
# print("Done!")

# torch.save(model.state_dict(), "data/model.pth")
# print("Saved PyTorch Model State to model.pth")

# model = NeuralNetwork()
# model.load_state_dict(torch.load("data/model.pth"))