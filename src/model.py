import torch.nn as nn

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1), # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1), # 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), # 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1), # 8 -> 4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), # 8 -> 4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 17, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x  = self.decoder(x1)
        return x1, x

class AE_base(nn.Module):
    def __init__(self):
        super(AE_base, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1), # 32 -> 16
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1), # 16 -> 8
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1), # 8 -> 4
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 17, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x  = self.decoder(x1)
        return x1, x