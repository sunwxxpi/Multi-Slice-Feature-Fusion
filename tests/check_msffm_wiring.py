"""use_msffm 이 팩토리 → 백본 → 모델까지 실제로 전달되고, fusion 이 올바른 스트림을 소비하는지 검증.

pvt_v2_bN.__init__ 이 **kwargs 를 super 로 넘기지 않으면 use_msffm 이 조용히 무시된다.
크래시가 나지 않으므로 NonLocalBlock 개수를 직접 세는 것 외에 확인할 방법이 없다.

fusion 검사는 CPU 에서 돈다. NonLocalBlock.W_z[1] 이 zero-init BatchNorm 이라 갓 만든 모델의
fusion 출력은 항등적으로 0 이고, 백본은 prev/next 채널에 대해 완전히 불변(delta 정확히 0.0)이다.
따라서 BN gamma 를 비-영으로 덮어쓰기 전에는 어떤 비교도 fusion 경로를 지나지 않는다.
"""
import torch
import torch.nn as nn

import networks.emcad.pvtv2 as pvtv2
from networks.emcad.networks import EMCADNet, EMCAD_SA_Net

# b0 는 stage3/4 채널이 160/256 이라 MSFFM(320/512 고정) 을 붙일 수 없다.
FACTORIES = ['pvt_v2_b0', 'pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5']
MSFFM_FACTORIES = FACTORIES[1:]


def count_msffm(module):
    return sum(1 for m in module.modules() if m.__class__.__name__ == 'NonLocalBlock')


# 1) 팩토리가 use_msffm 을 super 로 전달하는가
for name in FACTORIES:
    assert getattr(pvtv2, name)().use_msffm is False, f'{name}: 기본값이 False 가 아님'
for name in MSFFM_FACTORIES:
    assert getattr(pvtv2, name)(use_msffm=True).use_msffm is True, \
        f'{name}: use_msffm=True 가 무시됨 — __init__ 이 **kwargs 를 super 로 안 넘긴다'
# b0 는 채널 가드에 걸려야 한다. 걸린다는 것 자체가 인자가 super 까지 갔다는 증거다.
try:
    pvtv2.pvt_v2_b0(use_msffm=True)
except AssertionError:
    pass
else:
    raise AssertionError('pvt_v2_b0 가 MSFFM 채널 가드를 통과 — use_msffm 이 무시됐을 수 있다')

# 2) 백본이 실제로 NonLocalBlock 을 만드는가 (stage3/stage4 x prev/self/next = 6개)
assert count_msffm(pvtv2.pvt_v2_b2()) == 0
assert count_msffm(pvtv2.pvt_v2_b2(use_msffm=True)) == 6

# 3) 모델 레벨: baseline 은 0개, SA 는 6개
net_base = EMCADNet(num_classes=5, encoder='pvt_v2_b2', pretrain=False)
net_sa = EMCAD_SA_Net(num_classes=5, encoder='pvt_v2_b2', pretrain=False)
assert count_msffm(net_base) == 0, f'EMCADNet 에 MSFFM 이 붙었다: {count_msffm(net_base)}'
assert count_msffm(net_sa) == 6, f'EMCAD_SA_Net 의 MSFFM 개수가 6 이 아니다: {count_msffm(net_sa)}'

# 4) 모듈 이름 규약. 이름이 바뀌면 발표된 체크포인트가 strict=True 로 안 열리는데,
#    개수만 세는 검사는 이름 변경을 통과시킨다.
expected_names = {f'cross_attention_{w}_{s}' for s in (3, 4) for w in ('prev', 'self', 'next')} \
                 | {f'compress_{s}' for s in (3, 4)}
actual_names = set(dict(net_sa.backbone.named_children()))
missing = expected_names - actual_names
assert not missing, f'MSFFM 모듈 이름이 규약과 다르다 (누락: {sorted(missing)})'

# 5) forward 통과 + 출력 규약 (deep supervision 4단, 최종단이 마지막 원소)
for net, num_slices, label in [(net_base, 1, 'EMCADNet'), (net_sa, 3, 'EMCAD_SA_Net')]:
    net = net.cuda().eval()
    with torch.no_grad():
        P = net(torch.randn(2, num_slices, 512, 512).cuda())
    assert isinstance(P, list) and len(P) == 4, f'{label}: 출력이 4개 리스트가 아님 ({type(P)}, {len(P)})'
    assert tuple(P[-1].shape) == (2, 5, 512, 512), f'{label}: 최종단 shape {tuple(P[-1].shape)}'

# 6) fusion 이 prev/next 를 올바른 방향으로 소비하는가.
#    prev 채널만 바꾼 두 입력을 흘려, cross_attention_prev_3 의 key/value 입력은 바뀌고
#    cross_attention_next_3 의 것은 바뀌지 않아야 한다. prev/next 를 뒤바꾸면 정확히 반대가 된다.
net_sa = net_sa.cpu().eval()
for m in net_sa.modules():
    if type(m).__name__ == 'NonLocalBlock':
        nn.init.constant_(m.W_z[1].weight, 1.0)

cap = {'prev': [], 'next': [], 'outs': []}


def grab_other(key):
    # forward hook 의 input 은 (x_thisBranch, x_otherBranch) — key/value 쪽만 본다.
    return lambda mod, inp, out: cap[key].append(inp[1].detach().clone())


net_sa.backbone.cross_attention_prev_3.register_forward_hook(grab_other('prev'))
net_sa.backbone.cross_attention_next_3.register_forward_hook(grab_other('next'))
net_sa.backbone.register_forward_hook(
    lambda mod, inp, out: cap['outs'].append([t.detach().clone() for t in out]))

torch.manual_seed(1)
x_a = torch.randn(1, 3, 128, 128)
x_b = x_a.clone()
x_b[:, 0] = torch.randn(1, 128, 128)          # prev 채널만 교체
with torch.no_grad():
    net_sa(x_a)
    net_sa(x_b)

assert len(cap['prev']) == 2 and len(cap['next']) == 2, \
    f"fusion 이 호출되지 않았다 (prev {len(cap['prev'])}회, next {len(cap['next'])}회)"
# CPU 라 결정적이다. next 스트림은 채널 2 만의 함수이므로 비트 단위로 같아야 한다.
d_prev = (cap['prev'][0] - cap['prev'][1]).abs().max().item()
assert torch.equal(cap['next'][0], cap['next'][1]), \
    f'prev 채널을 바꿨는데 next attention 의 입력이 변했다 — prev/next 가 뒤바뀌었다 (delta_prev={d_prev:.3e})'
assert d_prev > 1e-3, f'prev 채널을 바꿨는데 prev attention 의 입력이 그대로다: {d_prev:.3e}'

# 7) fusion 결과가 실제로 x_main 에 반영되는가 (계산만 하고 버리면 파라미터가 학습되지 않는다).
#    측정 위치가 중요하다: 백본 stage3/4 출력에서는 delta 가 6.6e-02 / 1.4e-01 이지만
#    디코더를 지난 P[-1] 에서는 1.1e-07 까지 줄어 float 노이즈에 묻힌다.
outs_a, outs_b = cap['outs']
for stage in (2, 3):
    d = (outs_a[stage] - outs_b[stage]).abs().max().item()
    assert d > 1e-3, f'백본 stage{stage + 1} 출력이 prev 채널에 반응하지 않는다: {d:.3e}'

# 8) deep supervision 순서: 최종단(out_head1, scale 4) 이 마지막 원소여야 한다.
#    네 원소 모두 interpolate 뒤 (B,5,H,W) 라 shape 만으로는 순서를 구분할 수 없다.
for net, num_slices, label in [(net_base.cpu(), 1, 'EMCADNet'), (net_sa, 3, 'EMCAD_SA_Net')]:
    nn.init.constant_(net.out_head1.weight, 0)
    nn.init.constant_(net.out_head1.bias, 0)
    with torch.no_grad():
        P = net(torch.randn(1, num_slices, 128, 128))
    assert P[-1].abs().max().item() == 0, f'{label}: out_head1 을 0 으로 만들었는데 P[-1] 이 0 이 아니다'
    for i in (0, 1, 2):
        assert P[i].abs().max().item() > 0, f'{label}: P[{i}] 가 0 — deep supervision 순서가 뒤집혔다'

# 9) emcad_sa 는 pvt_v2_b0 를 거부해야 한다 (NonLocalBlock 채널이 320/512 고정)
try:
    EMCAD_SA_Net(num_classes=5, encoder='pvt_v2_b0', pretrain=False)
except AssertionError:
    pass
else:
    raise AssertionError('EMCAD_SA_Net 이 pvt_v2_b0 를 거부하지 않음')

print('OK: use_msffm 전달 + 모듈 이름 규약 + prev/next 방향 + fusion 반영 + DS 순서')
