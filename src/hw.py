import torch
import glob
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from data import *
from utils import *
from model import *
from cluster import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True)
parser.add_argument("--quality", required=True)
parser.add_argument("--trainX_path")
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--prediction_path")
args = parser.parse_args()

SEED = 0
SK_SEED = 0x5EED

EPOCH = 220
same_seeds(SEED)

trainX = np.load(args.trainX_path)
trainX_preprocessed = preprocess(trainX)

model = AE().cuda()
if args.quality == 'base':
    print('is base')
    model = AE_base().cuda()

if args.mode == 'train':

    img_dataset = Image_Dataset(trainX_preprocessed)

    criterion = nn.MSELoss()
    if args.quality == 'base':
        EPOCH = 100
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    
    model.train()
    n_epoch = EPOCH

    img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)

    for epoch in range(n_epoch):
        for data in img_dataloader:
            img = data
            img = img.cuda()

            output1, output = model(img)
            loss = criterion(output, img)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
        print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, loss.data))

    torch.save(model.state_dict(), args.checkpoint_path)

if args.mode == 'test':

    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()

    trainX = np.load(args.trainX_path)

    latents = inference(X=trainX, model=model)
    pred, X_embedded = predict(latents)

    save_prediction(invert(pred), args.prediction_path)