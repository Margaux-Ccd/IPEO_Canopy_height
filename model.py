import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# implement UNet following steps from this link: https://medium.com/@vipul.sarode007/u-net-unleashed-a-step-by-step-guide-on-implementing-and-training-your-own-segmentation-model-in-a38741776968
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encoder path
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder path - note the increased input channels due to concatenation
        self.up_conv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)  # 512 from up_conv4 + 512 from enc4
        
        self.up_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)   # 256 from up_conv3 + 256 from enc3
        
        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)   # 128 from up_conv2 + 128 from enc2
        
        self.up_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)    # 64 from up_conv1 + 64 from enc1
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder path with skip connections
        dec4 = self.up_conv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up_conv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up_conv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up_conv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)

# create and return Unet 12 input channels
def get_model(in_channels=12, out_channels=1):
    model = UNet(in_channels, out_channels)
    return model

# return Adam optimizer with learning rate - to be tested
def get_optimizer(model, learning_rate=0.0001, weight_decay=1e-4):
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)



'''# loss functon for continuous regression values
def get_loss_fn():
    return nn.SmoothL1Loss(reduction="mean")'''


def get_loss_fn():
    mse_loss = nn.MSELoss()
    l1_loss = nn.SmoothL1Loss()
    
    def combined_loss(pred, target):
        return 0.5 * mse_loss(pred, target) + 0.5 * l1_loss(pred, target)
    
    return combined_loss



