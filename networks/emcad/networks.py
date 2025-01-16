import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.emcad.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from networks.emcad.pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from networks.emcad.decoders import EMCAD


class EMCAD_SA_Net(nn.Module):
    def __init__(self, num_classes=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation='relu', encoder='pvt_v2_b2', pretrain=True):
        super(EMCAD_SA_Net, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        if encoder == 'pvt_v2_b0':
            self.backbone = pvt_v2_b0()
            path = './model/pvt/pvt_v2_b0.pth'
            channels=[256, 160, 64, 32]
        elif encoder == 'pvt_v2_b1':
            self.backbone = pvt_v2_b1()
            path = './model/pvt/pvt_v2_b1.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b2':
            self.backbone = pvt_v2_b2()
            path = './model/pvt/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b3':
            self.backbone = pvt_v2_b3()
            path = './model/pvt/pvt_v2_b3.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b4':
            self.backbone = pvt_v2_b4()
            path = './model/pvt/pvt_v2_b4.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b5':
            self.backbone = pvt_v2_b5() 
            path = './model/pvt/pvt_v2_b5.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'resnet18':
            self.backbone = resnet18(pretrained=pretrain)
            channels=[512, 256, 128, 64]
        elif encoder == 'resnet34':
            self.backbone = resnet34(pretrained=pretrain)
            channels=[512, 256, 128, 64]
        elif encoder == 'resnet50':
            self.backbone = resnet50(pretrained=pretrain)
            channels=[2048, 1024, 512, 256]
        elif encoder == 'resnet101':
            self.backbone = resnet101(pretrained=pretrain)  
            channels=[2048, 1024, 512, 256]
        elif encoder == 'resnet152':
            self.backbone = resnet152(pretrained=pretrain)  
            channels=[2048, 1024, 512, 256]
        else:
            print('Encoder not implemented! Continuing with default encoder pvt_v2_b2.')
            self.backbone = pvt_v2_b2()  
            path = './model/pvt/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
            
        if pretrain==True and 'pvt_v2' in encoder:
            save_model = torch.load(path)
            self.backbone.load_state_dict(save_model, strict=False)
        
        print('Model %s created, param count: %d' %
                     (encoder+' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))
        
        #   decoder initialization
        self.decoder = EMCAD(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, lgag_ks=lgag_ks, activation=activation)
        
        print('Model %s created, param count: %d' %
                     ('EMCAD decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
             
        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)
        
    def forward(self, x):
        x_prev = x[:, 0:1, :, :]   # (B,1,H,W)
        x_main = x[:, 1:2, :, :]   # (B,1,H,W)
        x_next = x[:, 2:3, :, :]   # (B,1,H,W)
        
        # if grayscale input, convert to 3 channels
        if x_prev.size()[1] == 1:
            x_prev = self.conv(x_prev)
        if x_main.size()[1] == 1:
            x_main = self.conv(x_main)
        if x_next.size()[1] == 1:
            x_next = self.conv(x_next)
        
        # encoder
        x1, x2, x3, x4 = self.backbone(x_prev, x_main, x_next)

        # decoder
        dec_outs = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        p4 = self.out_head4(dec_outs[0])
        p3 = self.out_head3(dec_outs[1])
        p2 = self.out_head2(dec_outs[2])
        p1 = self.out_head1(dec_outs[3])

        p4 = F.interpolate(p4, scale_factor=32, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=16, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=8, mode='bilinear')
        p1 = F.interpolate(p1, scale_factor=4, mode='bilinear')
        
        return [p4, p3, p2, p1]

        
if __name__ == '__main__':
    model = EMCAD_SA_Net().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    P = model(input_tensor)
    print(P[0].size(), P[1].size(), P[2].size(), P[3].size())