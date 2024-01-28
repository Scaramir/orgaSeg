"""
scaramir, 2022
date: 2022-11-08
This is just a prototype sketch and can be seen as a starting point for the implementation of a neural network with pytorch.
NOTE: WIP - not finished yet
"""

#-----------Hyperparameters-----------
use_normalize = True
learning_rate = 0.0005
batch_size = 32
num_epochs = 15
num_classes = 2
load_trained_model = True
pretrained = True  # transfer learning
reset_classifier_with_custom_layers = False
train_network = False
evaluate_network = True

# all three are residual networks with a different architecture (-> images from google for visualization)
model_type = 'resnet18'
#model_type = 'resnext50_32x4d'
# model_type = 'wide_resnet50_2'

pic_folder_path = '/path/to/your/data'

input_model_path = None # './../models'
input_model_name = None # "model_resnet_18"

output_model_path = './../models/'
#----------------------------------


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
from torchmetrics import F1Score
from torchvision import datasets, transforms
from captum.attr import LRP
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from tqdm import tqdm 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import random
import warnings
import time
import copy
from mo_nn_helpers import get_mean_and_std
from mo_nn_helpers import *

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        warnings.warn('CUDA not available. Using CPU instead.', UserWarning)
    print('Device set to {}.'.format(device))
    return device
device = get_device()

# set seeds for reproducibility
def set_seeds(device = 'cuda', seed = 1129142087):
    random.seed(seed)
    np.random.seed(seed+1)
    torch.random.manual_seed(seed+2)
    if device == 'cuda':
        torch.cuda.manual_seed(seed+3)
        torch.cuda.manual_seed_all(seed+4)
        torch.backends.cudnn.deterministic = True
    print('Seeds set to {}.'.format(seed))
    return
set_seeds(device)

data_dir = pic_folder_path
# data_dir = 'C:/Users/.../Project 2/data'

if use_normalize: 
    mean, std = get_mean_and_std(data_dir)

# Data augmentation and normalization for training
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224), antialias='warn'),
        transforms.RandomRotation(degrees=(-35, 35)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.4),
        transforms.ToTensor() # transforms.ToTensor() converts the image to a tensor with values between 0 and 1, should we move this to the beginning of the pipeline?
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]),
}
if use_normalize:
    data_transforms["train"].transforms.append(transforms.Normalize(mean=mean, std=std, inplace=True))
    data_transforms["test"].transforms.append(transforms.Normalize(mean=mean, std=std, inplace=True))


# ---------------Data Loader------------------
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                    for x in ["train", "test"]}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                    shuffle=True, num_workers=0)
                    for x in ["train", "test"]}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
class_names = image_datasets["test"].classes
num_classes = len(class_names)
# --------------------------------------------


def get_model(model_type, load_trained_model, reset_classifier_with_custom_layers, num_classes=num_classes, pretrained=True, device='cuda', input_model_path=None, input_model_name=None):
    print('Loading model...')
    if (load_trained_model) & (input_model_path is not None) & (input_model_name is not None):
        model = load_model(input_model_path, input_model_name).to(device)
        print('Loaded model \'{}\'.'.format(input_model_name))
    else:
        # Load the pretrained model from pytorch
        if model_type == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif model_type == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif model_type == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        elif model_type == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
        elif model_type == 'vgg16_bn':
            model = models.vgg16_bn(pretrained=pretrained)
        elif model_type == 'vgg19':
            model = models.vgg19(pretrained=pretrained)
        elif model_type == 'vgg19_bn':
            model = models.vgg19_bn(pretrained=pretrained)
        elif model_type == 'densenet121':
            model = models.densenet121(pretrained=pretrained)
        elif model_type == 'inception_v3':
            model = models.inception_v3(pretrained=pretrained)
        elif model_type == 'googlenet':
            model = models.googlenet(pretrained=pretrained)
        elif model_type == 'shufflenet_v2_x0_5':
            model = models.shufflenet_v2_x0_5(pretrained=pretrained)
        elif model_type == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
        elif model_type == 'resnext50_32x4d':
            model = models.resnext50_32x4d(pretrained=pretrained)
        elif model_type == 'wide_resnet50_2':
            model = models.wide_resnet50_2(pretrained=pretrained)
        elif model_type == 'mnasnet0_75':
            model = models.mnasnet0_75(pretrained=pretrained)
        else:
            print('Model type not found.')
            return None

    if reset_classifier_with_custom_layers:
        # TODO: change access to in_features to replace classifier in case a model requires it
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 256),
                                    nn.Dropout(p=0.4, inplace=True),
                                    nn.Linear(256, 100),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(100, num_classes))
        #model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, 256),
        #                            nn.Dropout(p=0.4, inplace=True),
        #                            nn.Linear(256, 100),
        #                            nn.ReLU(inplace=True),
        #                            nn.Linear(100, num_classes))
    model = model.to(device)
    print("Done.")
    return model
model = get_model(model_type=model_type, load_trained_model=load_trained_model, reset_classifier_with_custom_layers=reset_classifier_with_custom_layers, num_classes=num_classes, pretrained=pretrained, device=device, input_model_path=input_model_path, input_model_name=input_model_name)

criterion = nn.CrossEntropyLoss()
# SGD optimizer with momentum could lead faster to good results, but Adam is more stable
optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, writer, save_model=True, output_model_path=output_model_path):
    best_accuracy = 0.0
    epoches_used = 0
    patience = 5
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        train_loss, train_accuracy, all_predictions, all_labels = train_loop(model, train_loader, criterion, optimizer)
        scheduler.step()
        val_loss, val_accuracy, all = validate_model(model, test_loader, criterion)
        writer.add_scalar('Loss/Train', train_loss, global_step=epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, global_step=epoch)
        writer.add_scalar('Loss/Validation', val_loss, global_step=epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, global_step=epoch)
        # Patience stopping
        if train_accuracy > val_accuracy:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
        else:
            epochs_without_improvement = 0
            if val_accuracy >= best_accuracy:
                best_accuracy = val_accuracy
                # save model
                save_model(model, output_model_path, "model_{}_{}".format(model_type, epoch))

        epoches_used += 1
    return best_accuracy, epoches_used, all_predictions, all_labels

def train_loop(model, train_loader, criterion, optimizer):
    # no cache clearing necessary since we're not using a GPU
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    # train loop
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    # returns
    train_loss /= len(train_loader)
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy

def validate_model(model, test_loader, criterion):
    val_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # store results for later use (confusion matrix, classification report)
            if 'all_predicted' in locals():
                all_predicted = torch.cat((all_predicted, predicted), 0)
                all_labels = torch.cat((all_labels, labels), 0)
            else:
                all_predicted = predicted
                all_labels = labels
            
    # returns
    val_loss /= len(test_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy, all_predicted, all_labels

# Perform hyperparameter grid search and save results to TensorBoard
# train only for 5 epochs to save time
# TODO: pass hyperparams as dict or class object
def hyperparameter_search(model_types, learning_rates, batch_sizes, train_dataset, test_dataset, num_epochs):
    # tqdm magic to update bars    
    total_combinations = len(learning_rates) * len(batch_sizes) * len(model_types)
    pbar = tqdm(total=total_combinations, desc='Hyperparameter Search')
    # This might take a while ... 
    writer = SummaryWriter()
    best_model_settings = {"global_best_accuracy": 0.0}
    for model_type in model_types: 
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                set_seeds(123420)
                torch.cuda.empty_cache()
                # Define the data loaders
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size)

                # Create an instance of the model for each parameter combination
                model = get_model(model_type=model_type, load_trained_model=load_trained_model, reset_classifier_with_custom_layers=reset_classifier_with_custom_layers, num_classes=num_classes, pretrained=pretrained, device=device, input_model_path=input_model_path, input_model_name=input_model_name)

                # Define the loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                # TODO: load and reset optimizer
                # TODO: reset passed scheduler
                scheduler = scheduler

                # Train the model
                best_accuracy, epoches_used, _, _ = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, writer, save_model=False)

                # Save parameter combination and best accuracy to TensorBoard
                writer.add_hparams({'model_type': model_type,
                                    'learning_rate': learning_rate,
                                    'batch_size': batch_size,
                                    'epoches_used': epoches_used},
                                    {'Best Accuracy': best_accuracy})

                if best_accuracy >= best_model_settings["global_best_accuracy"]:
                    # save settings for best model
                    best_model_settings = {
                        'global_best_accuracy': best_accuracy,
                        'model_type': model_type,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'epoches_used': epoches_used
                    }

                # tqdm magic to update bars
                pbar.update(1)
    writer.close()
    return best_model_settings

def train_best_model(best_model_settings, train_dataset, test_dataset, num_epochs):
    # Define the data loaders
    train_loader = DataLoader(train_dataset, batch_size=best_model_settings['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_model_settings['batch_size'])

    # Create an instance of the model for each parameter combination
    model = get_model(model_type=best_model_settings['model_type'], load_trained_model=load_trained_model, reset_classifier_with_custom_layers=reset_classifier_with_custom_layers, num_classes=num_classes, pretrained=pretrained, device=device, input_model_path=input_model_path, input_model_name=input_model_name)

    # Exponential learning rate scheduler
    scheduler = scheduler

    # Train the model
    writer = SummaryWriter()
    best_accuracy, epoches_used, all_predictions, all_labels = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, writer)
    f1_score = F1Score(all_labels, all_predictions, average='macro')
    writer.add_hparams(best_model_settings, {'F1-Score': f1_score})
    writer.close()

    # confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    # classification report
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    return


def predict_folder(trained_model, image_loader):
    # this works similar to validate_model, but it returns the predictions instead of the accuracy, since we don't have labels
    trained_model.eval()
    with torch.no_grad():
        for inputs, _ in image_loader:
            inputs = inputs.to(device)
            outputs = trained_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # store results for later use (confusion matrix, classification report)
            if 'all_predicted' in locals():
                all_predicted = torch.cat((all_predicted, predicted), 0)
            else:
                all_predicted = predicted
    # save prediction and image name to csv
    df = pd.DataFrame({'image_name': image_loader.dataset.samples, 'label': all_predicted})
    df.to_csv('predictions.csv', index=False)
    return
