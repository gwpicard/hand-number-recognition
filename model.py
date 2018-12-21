# models.py
# PyTorch model functions

from collections import OrderedDict
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, datasets, models
import torch.nn.functional as F

def load_model(filepath=None):
    '''
    Load model base layer for inference and import state parameters
    from checkpoint saved after model training
    '''
    # import resnet18 base layer
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features

    # create custom classifier layer on top of Resnet18
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_ftrs, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.3)),
        ('fc2', nn.Linear(4096, 6)), # for the 6 possible categories
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # replace resnet18 output classifying layer
    model.fc = classifier

    if filepath:
        # load checkpoint parameters from training
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # define optimiser
    optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

    return model, loss_function, optimiser

def train_model(model, loss_function, optimiser, train_dataset, valid_dataset, epochs=None):
    # Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)

    # get GPU device if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define loss function - CrossEntropyLoss
    loss_function = nn.CrossEntropyLoss()

    # define Adam for fully-connected layer of model only (momentum-based)
    optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.5)

    # send model to CPU/GPU
    model.to(device)

    if epochs is None:
        epochs = 1

    print("Training model for %d epochs"%epochs)

    update_steps = 2

    loss_hist = []

    # train model
    for e in range(epochs):
        model.train()
        steps = 0

        # tracking metrics for training
        n_data = 0
        running_loss = 0
        train_accuracy = 0
        for data, labels in trainloader:
            steps+=1

            # send data to GPU
            data, labels = data.to(device), labels.to(device)

            # feed forward data + calculate loss
            output = model.forward(data)
            loss = loss_function(output, labels)

            # backprop + gradient descent
            loss.backward()
            optimizer.step()

            # track training running loss
            running_loss += loss.item()*data.size(0)

            # track train accuracy counter
            _, pred = torch.max(output, 1)
            n_correct = torch.sum(labels.data == pred)
            train_accuracy += n_correct.double()

            # keep track of number of data points for averaging
            n_data += data.size(0)

            if steps % update_steps == 0:
                # track validation loss + accuracy
                valid_loss = 0
                valid_accuracy = 0

                model.eval() # evaluation mode
                with torch.no_grad():
                    for data_v, labels_v in validloader: # iterate through validation data

                        data_v, labels_v = data_v.to(device), labels_v.to(device)
                        output = model.forward(data_v)
                        v_loss = loss_function(output, labels_v)

                        # update validation loss
                        valid_loss += v_loss.item()*data_v.size(0)

                        # calculate valid accuracy for batch
                        _, pred = torch.max(output, 1)
                        n_correct = torch.sum(labels_v.data == pred)
                        valid_accuracy += n_correct.double()

                print("Epoch %d/%d.. Steps %d/%d.. "%(e+1, epochs, steps, len(trainloader)))
                train_str = "Training loss:  %.3f.. Training accuracy: %.3f.."%(running_loss/n_data, train_accuracy/n_data)
                val_str = "Validation loss:  %.3f.. Validation accuracy: %.3f.."%(valid_loss/len(validloader.dataset), valid_accuracy/len(validloader.dataset))
                print(train_str, val_str)

                # keep track of loss with training
                loss_hist.append([running_loss/n_data, valid_loss/len(validloader.dataset)])

                # reset tracking metrics
                running_loss = 0
                train_accuracy = 0
                n_data = 0

                model.train()

    checkpoint = {'model_state_dict':  model.state_dict(),
                  'optimizer_state_dict': optimiser.state_dict()}

    return checkpoint

def predict_label(model, image):
    model.eval();
    with torch.no_grad():
        output = model.forward(image)

        ps = torch.exp(output)
        probs, idx = ps.topk(6)

        # probabilities for each category
        probs = probs.detach().numpy().flatten()
        idx = (idx.detach().numpy()+1).flatten()

        predict = sorted(zip(idx, probs), key=lambda x:x[0])


    return predict # return all labels
    # return idx[0] # return top label
