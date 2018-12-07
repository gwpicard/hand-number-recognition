# models.py
# PyTorch model functions

from collections import OrderedDict
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, datasets, models
import torch.nn.functional as F

def load_model(filepath):
    '''
    Load model base layer for inference and import state parameters
    from checkpoint saved after model training
    '''
    # import resnet18 base layer
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features

    # create custom classifier layer on top of Resnet18
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_ftrs, 1024)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.3)),
        ('fc2', nn.Linear(1024, 6)), # for the 6 possible categories
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # replace resnet18 output classifying layer
    model.fc = classifier

    # load checkpoint parameters from training
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    # set model to eval
    model.eval()

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # define optimiser
    optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

    return model, loss_function, optimiser

def train_model(model, loss_function, optimiser, train_dataset, valid_dataset, epochs=None):
    # Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=20, shuffle=True)

    # get GPU device if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # send model to GPU
    model.to(device)

    # set number of epochs if not selected
    if epochs is None:
        epochs = 1

    running_loss = 0
    update_steps = 1

    # train model
    for e in range(epochs):
        model.train()
        steps = 0
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

            # update running loss
            running_loss += loss.item()

            if steps % update_steps == 0:
                # track validation loss + accuracy
                valid_loss = 0
                validation_accuracy = 0
                model.eval() # evaluation mode

                with torch.no_grad():
                    for data_1, labels_1 in validloader: # iterate through validation data

                        data_1, labels_1 = data_1.to(device), labels_1.to(device)
                        output = model.forward(data_1)
                        v_loss = loss_function(output, labels_1)

                        # update validation loss
                        valid_loss += v_loss.item()*data_1.size(0)

                        # calculate accuracy for batch
                        _, pred = torch.max(output, 1)
                        n_correct = torch.sum(labels_1.data == pred)
                        validation_accuracy += n_correct.double()

                print("Epoch %d/%d.. Steps %d/%d.. "%(e+1, epochs, steps, len(trainloader)))
                print("Training loss:  %.3f.."%(running_loss/update_steps))
                print("Validation loss:  %.3f.. Validation accuracy: %.3f.."%(valid_loss/len(validloader.dataset), validation_accuracy/len(validloader.dataset)))
                running_loss = 0
                model.train()

    checkpoint = {'model_state_dict':  model.state_dict(),
                  'optimizer_state_dict': optimiser.state_dict()}

    return checkpoint

def predict_label(model, image):
    with torch.no_grad():
        output = model.forward(image)

        ps = torch.exp(output)
        probs, idx = ps.topk(6)

        # probabilities for each category
        probs = probs.detach().numpy().flatten()
        idx = (idx.detach().numpy()+1).flatten()

        predict = sorted(zip(idx, probs), key=lambda x:x[0])

    return predict
