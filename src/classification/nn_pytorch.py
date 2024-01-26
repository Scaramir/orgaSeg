"""
scaramir, 2022
date: 2022-11-08
This is just a prototype sketch and can be seen as a starting point for the implementation of a neural network with pytorch.
TODO: Adjust it to your needs. 
NOTE: This was meant for a binary classification task for university purpose only.
NOTE: intertwined with `mo_nn_helpers.py`
"""

#-----------Hyperparameters-----------
use_normalize = True
pic_folder_path = '/path/to/your/data'
learning_rate = 0.0005
batch_size = 32
num_epochs = 15
num_epochs = 5
num_classes = 2
load_trained_model = True
pretrained = True  # transfer learning
reset_classifier_with_custom_layers = False
train_network = False
evaluate_network = True

# all three are residual networks with a different architecture (-> images from google for visualization)
#model_type = 'resnet18'
#model_type = 'resnext50_32x4d'
model_type = 'wide_resnet50_2'

input_model_path = './../models'
input_model_name = "model_resnet_18"

output_model_path = './../models/'
output_model_name = 'model_wide_resnet_2'
#output_model_name = 'model_resnext50_2'
#output_model_name = 'model_resnet_18'
#----------------------------------


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import torchvision.models as models
from sklearn.metrics import confusion_matrix, classification_report, plot_roc_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

from tqdm import tqdm
from mo_nn_helpers import get_mean_and_std
from mo_nn_helpers import *

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        print('-----WARNING-----\nCUDA not available. Using CPU instead.')
    print('Device set to {}.'.format(device))
    return device
device = get_device()

# set seeds for reproducibility
def set_seeds(device = 'cuda', seed = 1129142087):
    random.seed(seed)
    np.random.seed(seed+1)
    torch.random.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    print('Seeds set to {}.'.format(seed))
    return
set_seeds(device)

data_dir = pic_folder_path
# data_dir = 'C:/Users/.../Project 2/data'

if use_normalize: 
    mean, std = get_mean_and_std(data_dir)

# Data augmentation and normalization for training
# TODO: add elastic deformation
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224), antialias='warn'),
        transforms.RandomRotation(degrees=(-20, 20)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
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
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

if train_network:
    train_nn(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, output_model_path, output_model_name, num_epochs=num_epochs)


# TODO: Evaluation of 3 different networks. Use softmax to get the probabilities for each of the binary classes.
# TODO: Use F1 score to compare the networks instead of accuracy in case of unbalanced data.
def evaluate_model(model, dataset_sizes, criterion, class_names, image_datasets, device="cuda", dataset = "test"):
    # for every image of our test set, we will prdict the class and the probability
    # save the probabilities and the classes in a list
    # save the ground truth classes in a list
    # calculate the loss and the accuracy
    # plot the confusion matrix with the older heatmap function from project 1. 
    # plot the ROC curve with the function from project 1
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                    shuffle=False, num_workers=0)
                    for x in [dataset]}

    num_samples = 0
    num_correct = 0
    true_labels_list = []
    pred_labels_list = []
    pred_scores_list = []
    file_names_list = []
    model.eval()
    
    if device == "cuda":
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloaders[dataset], desc="Evaluating the model...")):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            #scores = torch.sigmoid(outputs)
            scores = torch.nn.Softmax(dim=1)(outputs)

            pred_scores, pred_labels = torch.max(scores, 1)

            num_correct += torch.sum(pred_labels == labels.data)
            num_samples += pred_labels.size(0)
            true_labels_list.append(class_names[labels.cpu().detach().numpy()[0]])
            pred_labels_list.append(class_names[pred_labels.cpu().detach().numpy()[0]])
            pred_scores_list.append(pred_scores.cpu().detach().numpy()[0])
            file_names_list.append(image_datasets[dataset].imgs[i][0].split("/")[-1])

    accuracy = 100 * float(num_correct) / num_samples
    #loss = criterion(outputs, labels)
    print("Accuracy: {:.2f} %".format(accuracy))
    #print("Loss: {:.2f}".format(loss))

    print(classification_report(true_labels_list, pred_labels_list))
    conf_mat = confusion_matrix(true_labels_list, pred_labels_list)
    print(conf_mat)

    # plot the confusion matrix
    plt.clf()
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion matrix - " + input_model_name)
    plt.show()


    df = pd.DataFrame({
        "file_name": file_names_list,
        "true_label": true_labels_list,
        "pred_label": pred_labels_list,
        "pred_score": pred_scores_list})
    print("Done.")
    return df

if evaluate_network:
    df = evaluate_model(model, dataset_sizes, criterion, class_names, image_datasets, device="cuda", dataset = "test")

# TODO: Plot the results. (Also with a confusion matrix as heatmap?)

# TODO: report? 