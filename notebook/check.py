import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Ensure the src directory is in the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Importing the ResNet9 model and dataset loading functions
from resnet import ResNet9
import dataset

# Step 1: Load CIFAR-10 Data
dataset_name = 'CIFAR10'
train_set, eval_train_set, test_set, train_labels, max_val = dataset.load_dataset(dataset_name)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

# Step 2: Initialize the Model, Loss Function, Optimizer, and Scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet9(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Step 3: Train the Model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    scheduler.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader.dataset):.4f}")

# Function to evaluate the model
def evaluate(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Step 4: Evaluate the Model on Clean and Adversarial Test Sets
clean_acc = evaluate(test_loader, model)
print(f"Accuracy on clean test set: {clean_acc:.2f}%")

# Generate adversarial test set using the method from dataset.py
adversarial_test_set = dataset.DatasetWrapper(test_set, {}, mode='test_adversarial', corrupt_val=max_val)
adv_test_loader = DataLoader(adversarial_test_set, batch_size=128, shuffle=False, num_workers=2)

adv_acc = evaluate(adv_test_loader, model)
print(f"Accuracy on adversarial test set: {adv_acc:.2f}%")
