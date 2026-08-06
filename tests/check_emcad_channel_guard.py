"""EMCADNet 채널 가드 검증.

EMCADNet 은 설계상 1채널만 받는다 (`derive_num_slices('emcad', ...) == 1`). 가드가 없으면
forward 의 `if x.size()[1] == 1: x = self.conv(x)` 분기가 3채널 prev/reference/next
트리플렛을 그냥 통과시켜 RGB 처럼 소비한다 — 크래시 없이 결과만 조용히 틀려진다.
"""
import contextlib
import io

import torch

from networks.emcad.networks import EMCADNet


def build():
    with contextlib.redirect_stdout(io.StringIO()):  # 생성자의 param count print 억제
        return EMCADNet(num_classes=5, encoder='pvt_v2_b1', pretrain=False).to('cpu').eval()


model = build()

# 정상 경로: 설계대로의 1채널 입력은 여전히 통과해야 한다.
x1 = torch.randn(1, 1, 64, 64)
with torch.no_grad():
    out = model(x1)
assert len(out) == 4, 'EMCADNet 정상 경로(1채널)가 깨짐'

# 회귀 대상: 3채널 트리플렛이 조용히 RGB 로 소비되면 안 된다.
x3 = torch.randn(1, 3, 64, 64)
try:
    with torch.no_grad():
        model(x3)
except (ValueError, AssertionError, RuntimeError):
    pass
else:
    raise AssertionError(
        'EMCADNet 이 3채널 입력을 거부하지 않음 '
        '(prev/reference/next 트리플렛이 RGB 로 오인될 위험)')

print('OK: EMCADNet 이 3채널 입력을 거부하고 1채널 입력은 정상 통과')
