import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from smdebug import modes
from smdebug.pytorch import get_hook
import smdebug.pytorch as smd
from torchvision.io import read_image
import torchvision
import copy
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
import argparse
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(model, test_loader, loss_criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    print("Model testing has been started")    
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)        
        outputs=model(inputs)
        loss=loss_criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += float(loss.item() * inputs.size(0))
        running_corrects += float(torch.sum(preds == labels.data))

    loss = running_loss/(len(test_loader.dataset))
    accuracy = running_corrects/len(test_loader.dataset)
    
    print("Model testing has been completed") 
    
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

def train(model, train_loader, loss_criterion, optimizer, device, num_epochs):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    best_acc = 0.0
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predictions = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predictions == labels.data).item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, epoch_acc))
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_wts)
    return model

    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    
    model = models.resnet50(pretrained = True) 
    
    for param in model.parameters():
        param.requires_grad = False 
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential( nn.Linear( num_features, 256),
                              nn.ReLU(inplace = True),
                              nn.Linear(256, 133),
                              nn.ReLU(inplace = True) 
                            )
    return model

    

def create_data_loaders(data, batch_size, test_batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_path = os.path.join(data, "train")
    test_path = os.path.join(data, "test")
    
    training_transform = transforms.Compose([
        # transforms.Resize(255),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor() ])
    
    testing_transform = transforms.Compose([
        # transforms.Resize(225),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor() ])
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=training_transform)    
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=testing_transform)
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size )
    
    return train_data_loader, test_data_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=net()
    model = model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    
    train_data_loader, test_data_loader = create_data_loaders(args.data_path, args.batch_size, args.test_batch_size )
    
    loss_criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.learning_rate)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_data_loader, loss_criterion, optimizer, device, 1)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_data_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''

    parser=argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data_path', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument("--test_batch_size", type=int, default=1, metavar="N", help="input batch size for testing (default: 1000)" )
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR']) 
    args = parser.parse_args()
    
    
    main(args)