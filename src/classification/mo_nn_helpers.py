"""
scaramir, 2022
Utility functions for training and evaluating models.
Please rewrite this code and triple check it before using it for your own project. 
TODO: Adjust it to your needs. 
NOTE: This was meant for a binary classification task for university purpose only.
NOTE: intertwined with `nn_pytorch.py`
"""

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from tqdm import tqdm


def try_make_dir(d):
    '''
    Create Directory-path, if it doesn't exist yet.
    '''
    import os
    if not os.path.isdir(d):
        os.makedirs(d)
    return

def get_mean_and_std(data_dir):
    '''
    Acquire the mean and std color values of all images (RGB-values) in the training set.
    inpupt: "data_dir" string
    output: mean and std Tensors of size 3 (RGB)
    '''
    # Load the training set
    train_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transforms.ToTensor()) for x in ["train"]}
    train_loader = {x: torch.utils.data.DataLoader(dataset=train_dataset[x], batch_size=1, num_workers=0) for x in ["train"]}
    # Calculate the mean and std of the training set
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(train_loader["train"], desc="Calculating mean and std of all RGB-values"):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    #var[x] = E[x**2] - E[X]**2
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    print("Mean: ", mean, ", Std: ", std)
    return mean, std


# Save the whole model
def save_model(model, dir, model_name):
    torch.save(model, f"{dir}/{model_name}.pt")

# Load the whole model
def load_model(model_path, model_name):
    model = torch.load(f"{model_path}/{model_name}.pt")
    print(f"Loaded model {model_name}")
    return model


# ----------------------------------Plots-------------------------------
# These functions will be used in the train_nn function
def plot_loss(train_losses, test_losses, output_model_name, output_model_path):
    import matplotlib.pyplot as plt
    '''Plot the loss per epoch of the training and test set'''
    plt.clf()
    plt.plot(train_losses, "-o")
    plt.plot(test_losses, "-o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Test"])
    plt.title("Train vs. Test Loss")
    plt.savefig(
        f"{output_model_path}/train_test_loss_{output_model_name}.png")
    return

def plot_accuracy(train_accus, test_accus, output_model_name, output_model_path):
    '''Plot the accuracy per epoch of the training and test set'''
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(train_accus, "-o")
    plt.plot(test_accus, "-o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(["Train", "Test"])
    plt.title("Train vs. Test Accuracy")
    plt.savefig(
        f"{output_model_path}/train_test_accuracy_{output_model_name}.png")
    return
# ---------------------------------------------------------------------

# -------------------------------Training ResNet and ResNeXt-------------------------------
# FIXME: save state dict not model
# TODO: Switch to TensorBoard for logging and visualization 
#       - make sure to train each combination of Hyperparameters if they are given as Lists. Add the grid search to tensorboard log.
# TODO: include stopping criterion (patience stopping (5 epochs without improvement))
# OPTIONAL: include mixed precision training (automatic mixed precision)
# OPTIONAL: switch to torchmetrics.Accuracy() for accuracy calculation
# Advanced: use distributed training for better cluster usage (torch.nn.parallel.DistributedDataParallel)
# Advanced: DeepSpeed and Fabric?
def train_nn(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, output_model_path, output_model_name, num_epochs=25):
    import time
    import copy
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    try_make_dir(output_model_path)
    since = time.time()
    # Keep track of accuracy and loss
    best_model_wts = copy.deepcopy(model.state_dict())
    train_losses = []
    train_accus = []
    test_losses = []
    test_accus = []
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)
                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, pred_labels = torch.max(outputs, 1)        # only checking for correct predictions / label with highest prediction value
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                # track stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred_labels == labels.data)

            # stats for epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = 100. * running_corrects.double() / dataset_sizes[phase]
            if phase == "train":
                train_losses.append(epoch_loss)
                train_accus.append(epoch_acc.item())
            else:
                test_losses.append(epoch_loss)
                test_accus.append(epoch_acc.item())
            print('{} Loss: {:.3f} Acc: {:.4f}%'.format(phase, epoch_loss, epoch_acc))

            # deep copy the best model
            if phase == "test" and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_model(model, output_model_path, output_model_name)
                print(">> Model saved as: ", output_model_name)
                print(">> Model saved in: ", output_model_path)
        print()
        # update schedular (learning rate) after each epoch, instead of after each batch
        scheduler.step()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    print("----Done----")

    # Plot training curves and save the model
    plot_loss(train_losses, test_losses, output_model_name, output_model_path)
    plot_accuracy(train_accus, test_accus, output_model_name, output_model_path)
    save_model(model, output_model_path, output_model_name)
    print(">> Model saved as: ", output_model_name)
    print(">> Model saved in: ", output_model_path)

    return model
# ----------------------------------------------------------------------