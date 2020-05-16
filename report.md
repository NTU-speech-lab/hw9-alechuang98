<center><font size="30"><b>ML HW9</b></font></center>

<center><span style="font-weight:light; color:#7a7a7a; font-family:Merriweather;">by b06902034 </span><span style="font-weight:light; color:#7a7a7a; font-family:Noto Serif CJK SC;">黃柏諭</span></center>

---

### Problem 1

|             | Base                                                         | Improve                                                      |
| ----------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Autoencoder | conv(3, 64)<br/>ReLU()<br/>MaxPool2d(2)<br/>conv(64, 128)<br/>ReLU()<br/>MaxPool2d(2)<br/>conv(128, 256)<br/>ReLU()<br/>MaxPool2d(2) | conv(3, 64)<br/>BatchNorm2d(64)<br/>ReLU()<br/>MaxPool2d(2)<br/>conv(64, 128)<br/>BatchNorm2d(128)<br/>ReLU()<br/>conv(128, 128)<br/>BatchNorm2d(128)<br/>ReLU()<br/>MaxPool2d(2)<br/>conv(128, 256)<br/>BatchNorm2d(256)<br/>ReLU()<br/>conv(256, 256)<br/>BatchNorm2d(256)<br/>ReLU()<br/>MaxPool2d(2)<br/>conv(256, 512)<br/>BatchNorm2d(256)<br/>ReLU()<br/>MaxPool2d(2) |
| Epoch       | 100                                                          | 220                                                          |
| optimizer   | Adam                                                         | AdamW                                                        |
| scheduler   | constant                                                     | lr_scheduler.StepLR()                                        |
| Accuracy    | 0.623                                                        | 0.801                                                        |
| Embedding   | ![](/home/alec/Documents/ML/hw9-alechuang98/img/p1_baseline.png) | ![](/home/alec/Documents/ML/hw9-alechuang98/img/p1_improved.png) |
| clustering  | PCA: 4096 -> 256<br/>TSNE: 256 -> 2<br/>MiniBatchKMeans()    | PCA: 2048 -> 256<br/>TSNE: 256 -> 2<br/>MiniBatchKMeans()    |

* Autoencoder: 有嘗試過使用VAE，儘管會使embedding分佈較好，但是對training set在kaggle上的result沒有幫助，推測試因為本題目的testing set就是training set，並不會有新的未看過的圖片。加深深度其實就形同於加大的conv martix，eg. 兩層3\*3可以做到一層5\*5的效果，可以抽取較大的feature。
* Optimizer: AdamW好一些些
* LR scheduler: 加速訓練用

### Problem 2

![](/home/alec/Documents/ML/hw9-alechuang98/img/p2_result.png)

增加了BatchNorm layer後看上去某些pixel會有些顏色失真，但細節表現看起來較明顯，整體loss是變小的。

### Problem 3

![](/home/alec/Documents/ML/hw9-alechuang98/img/p3_result.png)

Accuracy的浮動很大，推測是因為沒有使用VAE導致很多沒看過的圖片沒辦法好好的抽出要使用的feature。

Loss穩定下降，大致符合預期。

