import argparse
import torch
from utils import load_data
from model import load_model, train_model

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', action="store")
parser.add_argument('--save_dir', action="store", dest="save_directory")
parser.add_argument('--epochs', action="store", dest="epochs", type=int)

args = parser.parse_args()

# load data
data_dir = args.data_dir
train_dataset, valid_dataset, test_dataset = load_data(data_dir)

# create model, loss function and optimiser
model, loss_function, optimiser = load_model()

# train model - return checkpoint
checkpoint = train_model(model, loss_function, optimiser, train_dataset, valid_dataset, args.epochs)

# save model after training
if args.save_directory:
    torch.save(checkpoint, args.save_directory+"/checkpoint.pth")
else:
    torch.save(checkpoint, 'checkpoint.pth')
