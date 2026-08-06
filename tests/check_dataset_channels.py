"""COCAVolumeDataset 의 num_slices 동작 검증.

1채널 모드가 3채널 모드의 center 와 픽셀 단위로 같아야 baseline 비교가 성립한다.
"""
import sys
import torch
from torchvision import transforms as T

from dataset import COCAVolumeDataset, load_hu_stats, Resize, ToTensor

ROOT = 'data/datasets/COCA/COCA_3frames_5fold'
N = 16

hu = load_hu_stats(f'{ROOT}/hu_stats_433.json')
samples = [l.strip() for l in open(f'{ROOT}/lists_COCA_5fold/fold0.txt') if l.strip()][:N]
tf = T.Compose([Resize(output_size=[512, 512]), ToTensor()])

def build(num_slices):
    return COCAVolumeDataset(f'{ROOT}/images', f'{ROOT}/labels', samples,
                             transform=tf, hu_stats=hu, num_slices=num_slices)

d3, d1 = build(3), build(1)

assert tuple(d3[0]['image'].shape) == (3, 512, 512), tuple(d3[0]['image'].shape)
assert tuple(d1[0]['image'].shape) == (1, 512, 512), tuple(d1[0]['image'].shape)
assert tuple(d3[0]['label'].shape) == (512, 512), tuple(d3[0]['label'].shape)

for i in range(N):
    assert torch.equal(d3[i]['image'][1:2], d1[i]['image']), f'{samples[i]}: center 채널 불일치'
    assert torch.equal(d3[i]['label'], d1[i]['label']), f'{samples[i]}: label 불일치'
    assert d3[i]['case_name'] == d1[i]['case_name'] == samples[i]

try:
    build(2)
except AssertionError:
    pass
else:
    print('FAIL: num_slices=2 가 거부되지 않음', file=sys.stderr)
    sys.exit(1)

print(f'OK: num_slices 1/3 형태 일치, {N} 샘플에서 center 채널·label 동일, 잘못된 값 거부')
