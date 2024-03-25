"""
scaramir, 2022
date: 2022-11-08
This is just a prototype sketch and can be seen as a starting point for the implementation of a neural network with pytorch.
NOTE: WIP - not finished yet
"""

import sklearn.metrics
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
from mo_nn_helpers import *
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Neural Network Training')
    parser.add_argument('--use_normalize', type=bool, default=True, help='Whether to use normalization')
    parser.add_argument('--learning_rate', type=float, default=[0.01, 0.005, 0.0005], nargs='+', help='Learning rate(s) for training')
    parser.add_argument('--batch_size', type=int, default=[32, 16], nargs='+', help='Batch size(s) for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes')
    parser.add_argument('--load_trained_model', type=bool, default=False, help='Whether to load a trained model')
    parser.add_argument('--reset_classifier_with_custom_layers', type=bool, default=True, help='Whether to reset the classifier with custom layers')
    parser.add_argument('--train_network', type=bool, default=True, help='Whether to train the network')
    parser.add_argument('--infere_folder', type=bool, default=True, help='Whether to evaluate the network')
    parser.add_argument('--model_type', type=str, default=['resnet18', 'resnext50_32x4d'], nargs='+', help='Type(s) of the model(s) to use for training')
    parser.add_argument('--pretrained', type=bool, default=True, help='Whether to use pretrained weights')
    parser.add_argument('--pic_folder_path', type=Path, default=Path('./../../data/data_sets/classification/'), help='Path to the picture folder')
    parser.add_argument('--input_model_path', type=Path, default=None, help='Path to the input model')
    parser.add_argument('--input_model_name', type=str, default=None, help='Name of the input model')
    parser.add_argument('--output_model_path', type=Path, default=Path('./../../models/'), help='Path to the output model')
    parser.add_argument('--hparam_seach', type=bool, default=True, help='Whether to perform hyperparameter search')
    
    # Get only known arguments
    known_args, _ = parser.parse_known_args()
    
    return known_args


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        warnings.warn('CUDA not available. Using CPU instead.', UserWarning)
    print('Device set to {}.'.format(device))
    return device

# set seeds for reproducibility
def set_seeds(device = 'cuda', seed = 12342069):
    random.seed(seed)
    np.random.seed(seed+1)
    torch.random.manual_seed(seed+2)
    if device == 'cuda':
        torch.cuda.manual_seed(seed+3)
        torch.cuda.manual_seed_all(seed+4)
        torch.backends.cudnn.deterministic = True
    print('Seeds set to {}.'.format(seed))
    return

# ---------------Data Loader------------------
def load_and_augment_images(pic_folder_path, batch_size, use_normalize=True):
    if use_normalize: 
        mean, std = get_mean_and_std(str(pic_folder_path))

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
            transforms.Resize((224, 224), antialias='warn'),
            transforms.ToTensor()
        ]),
    }
    if use_normalize:
        data_transforms["train"].transforms.append(transforms.Normalize(mean=mean, std=std, inplace=True))
        data_transforms["test"].transforms.append(transforms.Normalize(mean=mean, std=std, inplace=True))

    image_datasets = {x: datasets.ImageFolder(str(pic_folder_path / x), data_transforms[x])
                        for x in ["train", "test"]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                        shuffle=True, num_workers=0)
                        for x in ["train", "test"]}

    class_names = image_datasets["test"].classes
    num_classes = len(class_names)

    return dataloaders, class_names, num_classes
# --------------------------------------------


def get_model(model_type, load_trained_model, reset_classifier_with_custom_layers, num_classes=6, pretrained=True, device='cuda', input_model_path=None, input_model_name=None):
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


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, writer, save_model_to_disk=True, output_model_path="./../../models/", model_type="resnet18", device='cuda'):
    best_accuracy = 0.0
    epoches_used = 0
    patience = 5
    epochs_without_improvement = 0
    for epoch in tqdm(range(num_epochs), desc='Training', unit='epoch', leave=False):
        train_loss, train_accuracy, _, _ = train_loop(model, dataloaders, criterion, optimizer, device=device)
        scheduler.step()
        val_loss, val_accuracy, all_predictions, all_labels = validate_model(model, dataloaders, criterion, device=device)
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
                if save_model_to_disk:
                    save_model(model, output_model_path, "model_{}".format(model_type))

        epoches_used += 1
    return best_accuracy, epoches_used, all_predictions, all_labels

def train_loop(model, dataloaders, criterion, optimizer, device='cuda'):
    # no cache clearing necessary since we're not using a GPU
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    # train loop
    for inputs, labels in tqdm(dataloaders["train"], desc='Training', unit='batch', leave=False):
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
    train_loss /= len(dataloaders["train"])
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy, predicted, labels

def validate_model(model, dataloaders, criterion, device='cuda'):
    val_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders["test"], desc='Validation', unit='batch', leave=False):
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
    val_loss /= len(dataloaders["test"])
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy, all_predicted, all_labels

# Perform hyperparameter grid search and save results to TensorBoard
# train only for 5 epochs to save time
# TODO: pass hyperparams as dict or class object
def hyperparameter_search(model_types, learning_rates, batch_sizes, num_epochs, pic_folder_path, use_normalize=True, num_classes=6, pretrained=True, device='cuda', input_model_path=None, input_model_name=None, load_trained_model=False, reset_classifier_with_custom_layers=True):
    print("Starting hyperparameter search...")
    if num_epochs < 5:
        print("Warning: Number of epochs is less than 5. This might not be enough to get meaningful results.")
    if num_epochs > 20:
        print("Warning: Number of epochs is greater than 20. It will be capped to 20 so find the best configuration faster. You can retrain the best model with more epochs later on if needed.")
        num_epochs = 20

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
                dataloaders, _, _ = load_and_augment_images(pic_folder_path, batch_size, use_normalize)

                # Create an instance of the model for each parameter combination
                model = get_model(model_type=model_type, load_trained_model=load_trained_model, reset_classifier_with_custom_layers=reset_classifier_with_custom_layers, num_classes=num_classes, pretrained=pretrained, device=device, input_model_path=input_model_path, input_model_name=input_model_name)

                # NOTE: technically hyperparameter, but we're sticking to those for now
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
                scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

                # Train the model
                best_accuracy, epoches_used, _, _ = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, writer, save_model_to_disk=False)

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

def train_best_model(best_model_settings, dataloaders, num_epochs):
    # Create an instance of the model for each parameter combination
    model = get_model(model_type=best_model_settings['model_type'], load_trained_model=load_trained_model, reset_classifier_with_custom_layers=reset_classifier_with_custom_layers, num_classes=num_classes, pretrained=pretrained, device=device, input_model_path=input_model_path, input_model_name=input_model_name)

    # NOTE: technically hyperparameter, but we're sticking to those for now
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=best_model_settings['learning_rate'])
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    # Train the model
    writer = SummaryWriter()
    best_accuracy, epoches_used, all_predictions, all_labels = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, writer)
    all_predictions = all_predictions.cpu().detach().numpy()
    all_labels = all_labels.cpu().detach().numpy()
    # Save parameter combination and best accuracy to TensorBoard
    f1_score_test = sklearn.metrics.f1_score(all_labels, all_predictions, average='weighted')
    writer.add_hparams(best_model_settings, {'F1-Score test': f1_score_test})
    writer.close()
    print('Best accuracy: {:.2f}%'.format(best_accuracy))
    print('Epoches used: {}'.format(epoches_used))

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


def predict_folder(trained_model, class_names, infere_folder, device='cuda'):
    # get mean and std for normalization
    mean, std = get_mean_and_std(str(infere_folder))

    # load the data with nearly no augmentations and normalization
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224), antialias='warn'),
        transforms.ToTensor()
    ])
    if use_normalize:
        image_transforms.transforms.append(transforms.Normalize(mean=mean, std=std, inplace=True))

    image_dataset = datasets.ImageFolder(infere_folder, image_transforms)
    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=0)

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
    image_names = image_loader.dataset.samples
    print(image_names)
    image_names = [str(Path(image_name[0]).stem) for image_name in image_names]
    all_predicted = all_predicted.cpu().detach().numpy()
    all_predicted = [class_names[prediction] for prediction in all_predicted]
    # save prediction and image name to csv
    df = pd.DataFrame({'image_name': image_names, 'label': all_predicted})
    df.to_csv('./../../data/results/predictions.csv', index=False)
    return


# ---------------Main------------------
if __name__ == '__main__':
    args = parse_args()
    use_normalize = args.use_normalize
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_classes = args.num_classes
    load_trained_model = args.load_trained_model
    reset_classifier_with_custom_layers = args.reset_classifier_with_custom_layers
    train_network = args.train_network
    infere_folder = args.infere_folder
    model_type = args.model_type
    pretrained = args.pretrained
    pic_folder_path = args.pic_folder_path
    input_model_path = args.input_model_path
    input_model_name = args.input_model_name
    output_model_path = args.output_model_path
    hparam_search = args.hparam_seach

    device = get_device()
    set_seeds(device)


    # Hyperparameter search
    if hparam_search:
        best_model_settings = hyperparameter_search(
            model_types=model_type,
            learning_rates=learning_rate,
            batch_sizes=batch_size,
            num_epochs=num_epochs,
            pic_folder_path=pic_folder_path,
            use_normalize=use_normalize,
            num_classes=num_classes,
            pretrained=pretrained,
            device=device,
            input_model_path=input_model_path,
            input_model_name=input_model_name,
            load_trained_model=load_trained_model,
            reset_classifier_with_custom_layers=reset_classifier_with_custom_layers
        )
        print('Best model settings: {}'.format(best_model_settings))

    # Train the model
    if train_network and hparam_search:
        set_seeds()
        # Load the data
        dataloaders, class_names, _ = load_and_augment_images(pic_folder_path, best_model_settings['batch_size'], use_normalize)
        train_best_model(best_model_settings, dataloaders, num_epochs)
    
    if train_network and not hparam_search:
        set_seeds()
        model_settings = {
            'model_type': model_type,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epoches_used': num_epochs
        }
        # Load the data
        dataloaders, class_names, _ = load_and_augment_images(pic_folder_path, model_settings['batch_size'], use_normalize)
        train_best_model(model_settings, dataloaders, num_epochs)


    # Evaluate the model
    if infere_folder:
        best_model_settings = {
            'model_type': model_type,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epoches_used': num_epochs
        }
        # Load the best model
        trained_model = load_model(output_model_path, "model_{}".format(best_model_settings['model_type']))
        # Predict the labels for the test set
        predict_folder(trained_model, class_names, infere_folder, device=device, use_normalize=use_normalize)
    
    print('Done.')
# --------------------------------------------
