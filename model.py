import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# implement UNet following steps from this link: https://medium.com/@vipul.sarode007/u-net-unleashed-a-step-by-step-guide-on-implementing-and-training-your-own-segmentation-model-in-a38741776968
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # downsampling with encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck - most compressed image
        self.bottleneck = self.conv_block(512, 1024)

        # upsampling with decoder
        # increase spatial res
        self.upconv4 = self.upconv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    # 2 iterations of convulation and then ReLu activation function
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # not sure?
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # not sure? 
            nn.ReLU(inplace=True)
        )
    
    # Upsampling conv 
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2), # not sure?
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # not sure?
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoding
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoding
        up4 = self.upconv4(bottleneck)
        up3 = self.upconv3(up4 + enc4)
        up2 = self.upconv2(up3 + enc3)
        up1 = self.upconv1(up2 + enc2)

        # Output
        out = self.out_conv(up1 + enc1)
        return out

# create and return Unet 12 input channels
def get_model(in_channels=12, out_channels=1):
    model = UNet(in_channels, out_channels)
    return model

# return Adam optimizer with learning rate - to be tested
def get_optimizer(model, learning_rate=0.001):
    return optim.Adam(model.parameters(), lr=learning_rate)

# MSE loss functon for continuous regression values
def get_loss_fn():
    return nn.MSELoss() 


