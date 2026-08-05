"""use_msffm 이 팩토리 → 백본 → 모델까지 실제로 전달되는지 검증.

pvt_v2_bN.__init__ 이 **kwargs 를 super 로 넘기지 않으면 use_msffm 이 조용히 무시된다.
크래시가 나지 않으므로 NonLocalBlock 개수를 직접 세는 것 외에 확인할 방법이 없다.
"""
import torch

import networks.emcad.pvtv2 as pvtv2
from networks.emcad.networks import EMCADNet, EMCAD_SA_Net

# b0 는 stage3/4 채널이 160/256 이라 MSFFM(320/512 고정) 을 붙일 수 없다.
# 인자 전달 자체는 6개 모두 확인하되, 실제 블록 생성은 b1~b5 에서만 요구한다.
FACTORIES = ['pvt_v2_b0', 'pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5']


def count_msffm(module):
    return sum(1 for m in module.modules() if m.__class__.__name__ == 'NonLocalBlock')


# 1) 팩토리가 use_msffm 을 super 로 전달하는가
for name in FACTORIES:
    factory = getattr(pvtv2, name)
    assert factory().use_msffm is False, f'{name}: 기본값이 False 가 아님'
    assert factory(use_msffm=True).use_msffm is True, \
        f'{name}: use_msffm=True 가 무시됨 — __init__ 이 **kwargs 를 super 로 안 넘긴다'

# 2) 백본이 실제로 NonLocalBlock 을 만드는가 (stage3/stage4 x prev/self/next = 6개)
assert count_msffm(pvtv2.pvt_v2_b2()) == 0
assert count_msffm(pvtv2.pvt_v2_b2(use_msffm=True)) == 6

# 3) 모델 레벨: baseline 은 0개, SA 는 6개
net_base = EMCADNet(num_classes=5, encoder='pvt_v2_b2', pretrain=False)
net_sa = EMCAD_SA_Net(num_classes=5, encoder='pvt_v2_b2', pretrain=False)
assert count_msffm(net_base) == 0, f'EMCADNet 에 MSFFM 이 붙었다: {count_msffm(net_base)}'
assert count_msffm(net_sa) == 6, f'EMCAD_SA_Net 의 MSFFM 개수가 6 이 아니다: {count_msffm(net_sa)}'

# 4) forward 통과 + 출력 규약 (deep supervision 4단, 최종단이 마지막 원소)
for net, num_slices, label in [(net_base, 1, 'EMCADNet'), (net_sa, 3, 'EMCAD_SA_Net')]:
    net = net.cuda().eval()
    with torch.no_grad():
        P = net(torch.randn(2, num_slices, 512, 512).cuda())
    assert isinstance(P, list) and len(P) == 4, f'{label}: 출력이 4개 리스트가 아님 ({type(P)}, {len(P)})'
    assert tuple(P[-1].shape) == (2, 5, 512, 512), f'{label}: 최종단 shape {tuple(P[-1].shape)}'

# 5) emcad_sa 는 pvt_v2_b0 를 거부해야 한다 (NonLocalBlock 채널이 320/512 고정)
try:
    EMCAD_SA_Net(num_classes=5, encoder='pvt_v2_b0', pretrain=False)
except AssertionError:
    pass
else:
    raise AssertionError('EMCAD_SA_Net 이 pvt_v2_b0 를 거부하지 않음')

print('OK: use_msffm 이 팩토리·백본·모델까지 전달됨 (EMCADNet=0, EMCAD_SA_Net=6)')
