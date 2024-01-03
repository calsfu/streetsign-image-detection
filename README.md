### Traffic Sign Recognition with Convolutional Neural Network (CNN)

This repository contains a Convolutional Neural Network (CNN) implemented in PyTorch for the task of Traffic Sign Recognition. The model is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which consists of images of traffic signs belonging to 43 different classes.
Dataset

The GTSRB dataset is used for training and testing the model. It contains images of traffic signs with varying resolutions. The dataset is divided into training and testing sets. The training set is augmented with random transformations to increase the model's accuracy.

##Model Architecture

The CNN model consists of two convolutional layers followed by max-pooling layers. The flattened feature maps are then passed through three fully connected layers. The final layer uses the LogSoftmax activation function. The architecture is as follows:

    Convolutional Layer (input channels=3, output channels=8, kernel size=(5,5), stride=1)
    ReLU Activation
    MaxPooling Layer (kernel size=(2,2), stride=(2,2))
    Convolutional Layer (input channels=8, output channels=16, kernel size=(5,5), stride=1)
    ReLU Activation
    MaxPooling Layer (kernel size=(2,2), stride=(2,2))
    Fully Connected Layer (input size=4624, output size=800)
    ReLU Activation
    Fully Connected Layer (input size=800, output size=150)
    ReLU Activation
    Fully Connected Layer (input size=150, output size=numClasses)
    LogSoftmax Activation

##Training

The model is trained using the Adam optimizer with a learning rate of 1e-3. The CrossEntropyLoss is used as the loss function. Training is performed over 50 epochs, and both training and testing accuracies are printed during the training process.
##Results

After training, the model's state is saved to a file named "model.pth."
