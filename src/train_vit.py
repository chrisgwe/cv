import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import copy
from torch.utils.tensorboard import SummaryWriter

class BirdDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1])
        image = Image.open(img_path).convert("RGB")
        label = self.data_frame.iloc[idx, 0]

        if self.transform:
            image = self.transform(image)

        return image, label


train_csv = "/home/ziruiqiu/comp425/project/birds.csv"
val_csv = "/home/ziruiqiu/comp425/project/valid.csv"
test_csv = "/home/ziruiqiu/comp425/project/test.csv"

train_dir = "/home/ziruiqiu/comp425/project"
val_dir = "/home/ziruiqiu/comp425/project"
test_dir = "/home/ziruiqiu/comp425/project"

train_dataset = BirdDataset(train_csv, train_dir)
val_dataset = BirdDataset(val_csv, val_dir)
test_dataset = BirdDataset(test_csv, test_dir)

train_size = len(train_dataset)
val_size = len(val_dataset)
test_size = len(test_dataset)
print(f"Train size: {train_size}")
print(f"Val size: {val_size}")
print(f"Test size: {test_size}")
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

train_dataset.transform = data_transforms['train']
val_dataset.transform = data_transforms['val']
test_dataset.transform = data_transforms['test']

model_name = "vit_base_patch16_224"
num_classes = len(set(train_dataset.data_frame['labels']))
model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
#print(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Decay learning rate by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# scheduler = ReduceLROnPlateau(optimizer,'max' ,factor=0.1, patience=3)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

writer = SummaryWriter('experiments/vit_base_patch16_224_1')

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            if phase == 'train':
                train_loss = epoch_loss
            else:
                val_loss = epoch_loss

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Train the model
trained_model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=20)

# Save the model
torch.save(trained_model.state_dict(), '/home/ziruiqiu/comp425/project/models/vit_base_patch16_224_1.pth')

# Test the model
def test_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'Test loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')

test_model(trained_model, test_loader)