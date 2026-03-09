# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET

Image classification is a fundamental task in computer vision where an input image is assigned to one of several predefined classes. The objective of this experiment is to build and train a Convolutional Neural Network (CNN) using a labeled image dataset and evaluate its performance using accuracy, confusion matrix, and classification report.

## Neural Network Model
## DESIGN STEPS
### STEP 1: 

Load and Preprocess Data

### STEP 2: 

Get the shape of the first image in the training dataset

### STEP 3: 

Get the shape of the first image in the test dataset

### STEP 4: 

Train the Model

### STEP 5: 

Test the Model


### STEP 6: 

Predict on a Single Image and display the image



## PROGRAM

### Name:

### Register Number:

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0), -1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
          optimizer.zero_grad()
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()




        print('Name: KAVIYA V M')
        print('Register Number: 212224040154')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

### OUTPUT

## Training Loss per Epoch

<img width="652" height="207" alt="image" src="https://github.com/user-attachments/assets/6bda9a5c-ce71-4e54-8317-d2bceca8b0cc" />

## Confusion Matrix

<img width="812" height="692" alt="image" src="https://github.com/user-attachments/assets/846a1c6d-b993-4185-a8c7-0d6fa644c791" />


## Classification Report

<img width="596" height="415" alt="image" src="https://github.com/user-attachments/assets/8bde8392-a99d-4646-bbb1-50dbece19b98" />

### New Sample Data Prediction

<img width="532" height="612" alt="image" src="https://github.com/user-attachments/assets/676bb590-1a14-4098-a4ca-7e46354d4060" />

## RESULT

The Convolutional Neural Network (CNN) model was successfully trained and achieved good classification performance on the given image dataset.
