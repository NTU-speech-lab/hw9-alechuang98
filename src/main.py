import torch
import glob
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from data import *
from utils import *
from model import AE, AE_base
from cluster import *

TRAIN = 0
TEST = 0
HW = 1
EPOCH = 250
SEED = 0
SK_SEED = 0x5EED
same_seeds(SEED)

trainX = np.load('./data/trainX.npy')
trainX_preprocessed = preprocess(trainX)

MODEL = AE()

if TRAIN:
    img_dataset = Image_Dataset(trainX_preprocessed)

    model = MODEL.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

    model.train()
    n_epoch = EPOCH

    # 準備 dataloader, model, loss criterion 和 optimizer
    img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)

    # 主要的訓練過程
    for epoch in range(n_epoch):
        for data in img_dataloader:
            img = data
            img = img.cuda()

            output1, output = model(img)
            loss = criterion(output, img)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                torch.save(model.state_dict(), './checkpoints/checkpoint_{:03d}.pth'.format(epoch+1))
                
        print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, loss.data))

    # 訓練完成後儲存 model
    torch.save(model.state_dict(), './checkpoints_train/last_checkpoint.pth')


if TEST:
    # load model
    model = MODEL.cuda()
    model.load_state_dict(torch.load('./checkpoints_train/checkpoint_220.pth'))
    model.eval()

    # 準備 data
    trainX = np.load('./data/trainX.npy')

    # 預測答案
    latents = inference(X=trainX, model=model)
    pred, X_embedded = predict(latents)

    # 將預測結果存檔，上傳 kaggle
    save_prediction(pred, 'prediction.csv')

    # 由於是 unsupervised 的二分類問題，我們只在乎有沒有成功將圖片分成兩群
    # 如果上面的檔案上傳 kaggle 後正確率不足 0.5，只要將 label 反過來就行了
    save_prediction(invert(pred), 'prediction_invert.csv')

if HW:
    valX = np.load('./data/valX.npy')
    valY = np.load('./data/valY.npy')

    # ==============================================
    #  我們示範 basline model 的作圖，
    #  report 請同學另外還要再畫一張 improved model 的圖。
    # ==============================================
    model = MODEL.cuda()
    model.load_state_dict(torch.load('./checkpoints_hw/best.pth'))
    model.eval()
    latents = inference(valX, model)
    pred_from_latent, emb_from_latent = predict(latents, seed=SK_SEED)
    acc_latent = cal_acc(valY, pred_from_latent)
    print('The clustering accuracy is:', acc_latent)
    print('The clustering result:')
    plot_scatter(emb_from_latent, pred_from_latent, savefig='p1_label.png')
    plot_scatter(emb_from_latent, valY, savefig='p1_improved.png')

    plt.figure(figsize=(10,4))
    indexes = [1,2,3,6,7,9]
    imgs = trainX[indexes,]
    for i, img in enumerate(imgs):
        plt.subplot(2, 6, i+1, xticks=[], yticks=[])
        plt.imshow(img)
    # 畫出 reconstruct 的圖
    inp = torch.Tensor(trainX_preprocessed[indexes,]).cuda()
    latents, recs = model(inp)
    recs = ((recs+1)/2 ).cpu().detach().numpy()
    recs = recs.transpose(0, 2, 3, 1)
    for i, img in enumerate(recs):
        plt.subplot(2, 6, 6+i+1, xticks=[], yticks=[])
        plt.imshow(img)
    plt.tight_layout()
    plt.savefig('p2_result.png')
    plt.show()

    checkpoints_list = sorted(glob.glob('checkpoints/checkpoint_*.pth'))
    # load data
    dataset = Image_Dataset(trainX_preprocessed)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    points = []
    with torch.no_grad():
        for i, checkpoint in enumerate(checkpoints_list):
            print('[{}/{}] {}'.format(i+1, len(checkpoints_list), checkpoint))
            model.load_state_dict(torch.load(checkpoint))
            model.eval()
            err = 0
            n = 0
            for x in dataloader:
                x = x.cuda()
                _, rec = model(x)
                err += torch.nn.MSELoss(reduction='sum')(x, rec).item()
                n += x.flatten().size(0)
            print('Reconstruction error (MSE):', err/n)
            latents = inference(X=valX, model=model)
            pred, X_embedded = predict(latents)
            acc = cal_acc(valY, pred)
            print('Accuracy:', acc)
            points.append((err/n, acc))
    ps = list(zip(*points))
    plt.figure(figsize=(6,6))
    plt.subplot(211, title='Reconstruction error (MSE)').plot(ps[0])
    plt.subplot(212, title='Accuracy (val)').plot(ps[1])
    plt.savefig('p3_result.png')
    plt.show()