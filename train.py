#from utils import *
from dataset import *
from model import *
from loss import *
from config import parse_configs
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import math
from collections import OrderedDict
from torch.utils.data import Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger, TestTubeLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

from torch.optim import Adam
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import warnings

import torch.distributed as dist
from torch.utils.data.sampler import SubsetRandomSampler
import shutil
import pickle
import copy
import tqdm

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets



def get_acc(output, target):
    pred = torch.argmax(output, 1)
    acc = (pred == target).sum().item()
    return acc


# should use pytorch_lightning for parallel
def loss_batch(loss_func, output, target, opt = None):
    loss = loss_func(output, target)

    metric_b = get_acc(output, target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

def loss_epoch(model, loss_func, dataset_dl, device, opt = None):
    running_loss = 0.0
    running_metric = 0.0

    len_data = len(dataset_dl.dataset)

    for i, (img, label) in enumerate(tqdm(dataset_dl)):
        img = img.to(device)
        label = label.to(device)

        output = model(img)
        loss, metric_b = loss_batch(loss_func, output, label, opt)
        #update running loss
        running_loss += loss

        #update acc
        if metric_b is not None:
            running_metric += metric_b


    loss_epoch = running_loss/float(len_data)
    metric = running_metric/float(len_data)
    
    return loss_epoch, metric

def train_val(model, params, device):
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    path2weights=params["path2weights"]

    # history of loss values
    loss_history = {
            "train": [],
            "val" : []
            }

    acc_history = {
            "train" : [],
            "val" : []
            }


    #a deep copy of weights for the bes performing
    best_model_wts = copy.deepcopy(model.state_dict())

    #initialize best loss to a large value
    best_loss = float('inf')

    #main loop
    for epoch in range(num_epochs):
        # train model
        print("Start training with {}".format(epoch))
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, device, opt)

        #collect loss 
        loss_history["train"].append(train_loss)
        acc_history["train"].append(train_metric)

        #eval model
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, device)
            


        # Store best weight
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            # Store weight into a local file
            torch.save(model.state_dict(), path2weights)
        print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f" %(train_loss,val_loss, val_metric))
        
        #collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        acc_history["val"].append(val_metric)


    #load best model weights
    model.load_state_dict(best_model_wts)

    return model, loss_history, acc_history

if __name__ == '__main__':
    init_path = os.path.abspath('../')
    df = pd.read_csv(os.path.join(init_path,'train_new.csv'), sep = "\t")
    
    dataset = GLRDataset(df)
    
    batch_size = 8
    validation_split = 0.8
    dataset = train_val_dataset(dataset)

    train_dataset, val_dataset = dataset["train"], dataset["val"]

    
    # Creating PT data samplers and loaders:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle = True, drop_last = True)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                    shuffle = True, drop_last = True)
    
    # Define optimizer , loss && model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = parse_configs()

    model = Net(args)
    model = model.to(device)

    model = nn.DataParallel(model)
    
    criterion = ArcFaceLoss(args['argface.s'], args['argface.m'], crit = args["crit"])

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    params = {
        "num_epochs" : 10,
        "optimizer": optimizer,
        "loss_func" : criterion, 
        "train_dl": train_loader, 
        "val_dl": validation_loader,
        "path2weights": "../models/weights.pt"
    }

    model, loss_history, acc_history = train_val(model, params, device)
