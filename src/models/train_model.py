from pathlib import Path 
from typing import Union

#import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler


def load_split_train_valid(data_dir: str, valid_size = .2) -> \
    Union[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """loads the data from a given path

    Args:
        data_dir (str): path to data directory, including subfolders

        valid_size (float, optional): size of validation set. 
            Defaults to .2.

    Returns:
        Union[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
            iteratable train and validation sets
    """
    # define image transformations
    train_transforms = transforms.Compose([transforms.Resize(1200),
                                       transforms.ToTensor(),
                                       ])    
    valid_transforms = transforms.Compose([transforms.Resize(1200),
                                      transforms.ToTensor(),
                                      ])    
    
    # load image data and apply transformations
    train_data = datasets.ImageFolder(data_dir,       
                    transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir,
                    transform=valid_transforms)    

    # define split indices
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # define data loader 
    train_loader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=64)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                   sampler=valid_sampler, batch_size=64)

    return train_loader, valid_loader


if __name__ == "__main__":
    # load data  
    data_dir = Path(__file__).parents[2] / \
        "data" / "example_dataset" / "final"
    trainloader, testloader = load_split_train_valid(data_dir, 0.2)
    print(trainloader.dataset.classes)

    # maybe do data augmentation (flipping images etc.)
    
    # setup model
    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
    model = models.resnet50(pretrained=True)

    # freeze the pre-trained layers, so we don’t backprop through them during training
    for param in model.parameters():
        param.requires_grad = False

    # redefine fully connected layer        
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 10),
                                    nn.LogSoftmax(dim=1))

    # loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)

    # training settings
    epochs = 1
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []

    # train model
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # show validation
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss/len(trainloader))
                test_losses.append(test_loss/len(testloader))                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(testloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()