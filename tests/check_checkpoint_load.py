"""test.py 의 체크포인트 로딩 블록이 4개 decoder 의 실제 fold0 체크포인트를 받는지 검증.

블록을 옮겨 적지 않고 test.py 의 해당 소스 구간을 그대로 실행한다 — 복붙하면 prefix 게이팅이나
strict 를 바꿔도 테스트는 옛 사본을 계속 통과시킨다. 모델은 CPU 로 만든다 (GPU 불필요).
"""
import contextlib
import io
import os
import re
import sys
import textwrap
import types
from glob import glob

import torch
import segmentation_models_pytorch as smp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)  # 로딩 블록의 "./model/" 이 상대 경로다

from networks.emcad.networks import EMCADNet, EMCAD_SA_Net

# test.py 는 import 시점에 argparse 를 돌린다. fold 검증을 통과할 최소 인자를 준다.
sys.argv = ['test.py', '--exp_setting', 'fold0']
from test import add_encoder_prefix

with open(os.path.join(REPO, 'test.py'), encoding='utf-8') as f:
    SOURCE = f.read()

_m = re.search(r'\n(    snapshot_path = os\.path\.join\("\./model/".*?)\n    log_path = ',
               SOURCE, re.S)
assert _m, 'test.py 의 체크포인트 로딩 블록을 못 찾음 (snapshot_path ~ log_path 앵커 확인)'
LOAD_BLOCK = compile(textwrap.dedent(_m.group(1)), 'test.py:load_block', 'exec')

PARAMETER_PATH = 'epo300_bs16_lr1e-05'

# (decoder, encoder, exp_setting, 체크포인트 키 수)
CASES = [
    ('unet', 'resnet50_sa', 'msffm_resnet50_unet_fold0_seed42', 454),
    ('segformer', 'resnet50_sa', 'msffm_resnet50_segformer_fold0_seed42', 410),
    ('emcad', 'pvt_v2_b2', 'emcad_fold0_seed42', 563),
    ('emcad_sa', 'pvt_v2_b2', 'emcad_sa_fold0_seed42', 637),
]


def build(decoder, encoder):
    if decoder == 'unet':
        return smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=1, classes=5)
    if decoder == 'segformer':
        return smp.Segformer(encoder_name=encoder, encoder_weights=None, in_channels=1, classes=5)
    NetCls = EMCAD_SA_Net if decoder == 'emcad_sa' else EMCADNet
    with contextlib.redirect_stdout(io.StringIO()):  # EMCADNet 이 param count 를 찍는다
        return NetCls(num_classes=5, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                      add=True, lgag_ks=3, activation='relu6', encoder=encoder, pretrain=False)


def run_load_block(net, decoder, encoder, exp_setting):
    """test.py 의 로딩 블록을 실제 체크포인트 위에서 실행하고 고른 파일 경로를 돌려준다."""
    ns = {
        'os': os, 'glob': glob, 'torch': torch,
        'add_encoder_prefix': add_encoder_prefix,
        'args': types.SimpleNamespace(decoder=decoder),
        'net': net,
        'exp_path': os.path.join(f'{net.__class__.__name__}_{encoder}', 'COCA_512', exp_setting),
        'parameter_path': PARAMETER_PATH,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        exec(LOAD_BLOCK, ns)
    return ns['best_model_path']


for decoder, encoder, exp_setting, n_keys in CASES:
    net = build(decoder, encoder)
    path = run_load_block(net, decoder, encoder, exp_setting)
    assert os.path.isfile(path), path
    checkpoint = torch.load(path, map_location='cpu')
    assert len(checkpoint) == n_keys, f'{decoder}: 키 {len(checkpoint)}개 != {n_keys}개 ({path})'
    del net, checkpoint

# EMCAD 체크포인트에 encoder. 를 붙이면 모듈 트리와 어긋나 로드가 실패해야 한다.
# args.decoder 만 SMP 계열로 속여 같은 블록이 prefix 분기를 타게 한다.
# 이게 통과해 버리면 게이팅이 무의미해졌거나 strict=False 로 완화된 것이다.
for decoder, encoder, exp_setting, n_rewritten in [
        ('emcad', 'pvt_v2_b2', 'emcad_fold0_seed42', 347),
        ('emcad_sa', 'pvt_v2_b2', 'emcad_sa_fold0_seed42', 421)]:
    net = build(decoder, encoder)
    path = sorted(glob(os.path.join('./model/', f'{net.__class__.__name__}_{encoder}',
                                    'COCA_512', exp_setting, PARAMETER_PATH, '*_best_model.pth')))[0]
    checkpoint = torch.load(path, map_location='cpu')
    rewritten = sum(1 for k in add_encoder_prefix(checkpoint, prefix='encoder.') if k not in checkpoint)
    assert rewritten == n_rewritten, f'{decoder}: prefix 로 바뀌는 키 {rewritten}개 != {n_rewritten}개'
    try:
        run_load_block(net, 'unet', encoder, exp_setting)
    except RuntimeError:
        pass
    else:
        raise AssertionError(f'{decoder}: encoder. prefix 를 붙였는데 로드가 성공했다 '
                             f'(prefix 게이팅이 무의미해졌거나 strict=False)')
    del net, checkpoint

print(f'OK: test.py 로딩 블록이 decoder {len(CASES)}개의 fold0 체크포인트를 strict=True 로 로드, '
      f'EMCAD 계열에 prefix 를 붙이면 실패')
