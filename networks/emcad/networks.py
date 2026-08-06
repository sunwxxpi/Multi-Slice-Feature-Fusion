import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.emcad.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from networks.emcad.pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from networks.emcad.decoders import EMCAD


class EMCADNet(nn.Module):
    def __init__(self, num_classes=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation='relu', encoder='pvt_v2_b2', pretrain=True, use_msffm=False):
        super(EMCADNet, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        if encoder == 'pvt_v2_b0':
            self.backbone = pvt_v2_b0(use_msffm=use_msffm)
            path = './model/pvt/pvt_v2_b0.pth'
            channels=[256, 160, 64, 32]
        elif encoder == 'pvt_v2_b1':
            self.backbone = pvt_v2_b1(use_msffm=use_msffm)
            path = './model/pvt/pvt_v2_b1.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b2':
            self.backbone = pvt_v2_b2(use_msffm=use_msffm)
            path = './model/pvt/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b3':
            self.backbone = pvt_v2_b3(use_msffm=use_msffm)
            path = './model/pvt/pvt_v2_b3.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b4':
            self.backbone = pvt_v2_b4(use_msffm=use_msffm)
            path = './model/pvt/pvt_v2_b4.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b5':
            self.backbone = pvt_v2_b5(use_msffm=use_msffm) 
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
            self.backbone = pvt_v2_b2(use_msffm=use_msffm)  
            path = './model/pvt/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
            
        if pretrain==True and 'pvt_v2' in encoder:
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
        
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

        # EMCADNet 은 설계상 1채널만 받는다 (derive_num_slices('emcad', ...) == 1). 가드 없이
        # 3채널(prev/reference/next 트리플렛)이 들어오면 이 변환을 건너뛰고 RGB 처럼 소비해
        # 크래시 없이 결과만 조용히 틀려진다.
        if x.size()[1] != 1:
            raise ValueError(
                f'EMCADNet expects 1-channel input, got {x.size()[1]} channels')
        x = self.conv(x)

        # encoder
        x1, x2, x3, x4 = self.backbone(x)

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


class EMCAD_SA_Net(EMCADNet):
    """EMCAD + MSFFM. 입력 (B,3,H,W) 의 prev/reference/next 를 공유 conv 로 3채널화해 백본에 넘긴다.

    EMCADNet 과 반드시 별개 클래스로 유지할 것 — 체크포인트 경로가 `net.__class__.__name__`
    으로 합성되므로 하나로 합치면 두 실험이 서로 덮어쓴다.
    """
    def __init__(self, *args, **kwargs):
        encoder = kwargs.get('encoder', 'pvt_v2_b2')
        assert encoder in ('pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5'), \
            f"emcad_sa 는 pvt_v2_b1~b5 만 지원 (NonLocalBlock 채널이 320/512 고정): {encoder}"
        super().__init__(*args, use_msffm=True, **kwargs)

    def forward(self, x):
        x_prev = x[:, 0:1, :, :]   # (B,1,H,W)
        x_main = x[:, 1:2, :, :]
        x_next = x[:, 2:3, :, :]

        # 공유 conv 로 각 슬라이스를 1→3 채널화 (SMP `_sa` 인코더와 같은 가중치 공유 설계).
        if x_prev.size()[1] == 1:
            x_prev = self.conv(x_prev)
        if x_main.size()[1] == 1:
            x_main = self.conv(x_main)
        if x_next.size()[1] == 1:
            x_next = self.conv(x_next)

        x1, x2, x3, x4 = self.backbone(x_main, x_prev, x_next)

        dec_outs = self.decoder(x4, [x3, x2, x1])

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
    model = EMCADNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    P = model(input_tensor)
    print(P[0].size(), P[1].size(), P[2].size(), P[3].size())