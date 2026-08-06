"""train.py 의 SMP(unet/segformer) 모델 생성 블록이 --no_pretrain 을 실제로 반영하는지 검증.

블록을 옮겨 적지 않고 train.py 의 해당 소스 구간을 그대로 실행한다 — 복붙하면 하드코딩된
encoder_weights="imagenet" 을 고쳐도 테스트는 옛 사본을 계속 통과시킨다.
smp.Unet/Segformer 를 캡처용 더미로 바꿔치기해 실제 ImageNet 가중치 다운로드나 GPU 없이
factory 에 전달된 kwargs 만 검사한다 (`.cuda()` 는 더미가 self 를 반환).
"""
import importlib
import os
import re
import sys
import textwrap
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

with open(os.path.join(REPO, 'train.py'), encoding='utf-8') as f:
    SOURCE = f.read()

_m = re.search(
    r"\n(    if args\.decoder == 'unet':.*?pretrain=not args\.no_pretrain\)\.cuda\(\)\n)\n    # from torchinfo",
    SOURCE, re.S)
assert _m, 'train.py 의 모델 생성 블록을 못 찾음 (if args.decoder == unet ~ .cuda() 앵커 확인)'
MODEL_BLOCK = compile(textwrap.dedent(_m.group(1)), 'train.py:model_block', 'exec')


class _CaptureModel:
    """smp.Unet/Segformer 대역. kwargs 만 기록하고 실제 생성/다운로드는 하지 않는다."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def cuda(self):
        return self


def get_args(decoder, encoder, extra=()):
    sys.argv = ['train.py', '--exp_setting', 'no_pretrain_probe_fold0',
               '--decoder', decoder, '--encoder', encoder] + list(extra)
    if 'train' in sys.modules:
        train = importlib.reload(sys.modules['train'])
    else:
        import train
    return train.args


def run_model_block(train_args):
    fake_smp = types.SimpleNamespace(Unet=_CaptureModel, Segformer=_CaptureModel)
    ns = {'smp': fake_smp, 'args': train_args}
    exec(MODEL_BLOCK, ns)
    return ns['net'].kwargs


CASES = [
    ('unet', 'resnet50_sa'),
    ('segformer', 'mit_b2_sa'),
]

for decoder, encoder in CASES:
    args_default = get_args(decoder, encoder)
    kwargs_default = run_model_block(args_default)
    assert kwargs_default['encoder_weights'] == 'imagenet', (
        f'{decoder}: --no_pretrain 없이 encoder_weights={kwargs_default["encoder_weights"]!r} '
        f'(imagenet 이어야 함)')

    args_no_pretrain = get_args(decoder, encoder, extra=['--no_pretrain'])
    kwargs_no_pretrain = run_model_block(args_no_pretrain)
    assert kwargs_no_pretrain['encoder_weights'] is None, (
        f'{decoder}: --no_pretrain 지정했는데 encoder_weights={kwargs_no_pretrain["encoder_weights"]!r} '
        f'(None 이어야 함, ImageNet 가중치가 조용히 로드됨)')

print('OK: train.py 가 --no_pretrain 시 unet/segformer 모두 encoder_weights=None 을 SMP factory 에 전달')
