"""trainer.py 의 실제 손실 본문이 통합 전 두 계열의 손실을 그대로 재현하는지 검증.

기대값만 frozen main / EMCAD 의 식을 옮겨 적고, 실측값은 stub 모델·stub 로더로 trainer_coca 를
1 epoch 돌려 회수한다 — 식을 테스트에도 복붙하면 서로를 베끼는 동어반복이 되어 아무것도 못 잡는다.
"""
import contextlib
import io
import logging
import tempfile
import types

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn.modules.loss import CrossEntropyLoss

import trainer as trainer_mod
from utils import DiceLoss

B, C, HW = 2, 5, 16
TRAIN_TAG, VAL_TAG = 0, 1

dice_loss_class = DiceLoss()
ce_loss_class = CrossEntropyLoss()


def frozen_powerset(seq):
    """통합 전 EMCAD:utils.py 의 재귀 powerset 을 그대로 옮긴 것 (기대값 계산용 독립 기준)."""
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in frozen_powerset(seq[1:]):
            yield [seq[0]] + item
            yield item


def frozen_smp_loss(outputs, label_batch):
    """통합 전 main:trainer.py 의 손실식 (단일 출력, 0.5/0.5)."""
    with autocast():
        dice_loss = dice_loss_class(outputs, label_batch, softmax=True)
        ce_loss = ce_loss_class(outputs, label_batch)
        loss = (0.5 * dice_loss) + (0.5 * ce_loss)
    return dice_loss.item(), ce_loss.item(), loss.item()


def frozen_emcad_loss(P, label_batch, strategy):
    """통합 전 EMCAD:trainer.py 의 손실 루프 (deep supervision, 0.7/0.3)."""
    with autocast():
        out_idxs = list(range(len(P)))
        if strategy == 'mutation':
            ss = list(frozen_powerset(out_idxs))
        elif strategy == 'deep_supervision':
            ss = [[x] for x in out_idxs]
        else:
            ss = [[-1]]

        sum_dice_loss = 0.0
        sum_ce_loss = 0.0
        loss = 0.0
        for s in ss:
            if not s:
                continue
            iout = sum(P[idx] for idx in s)
            dice_loss = dice_loss_class(iout, label_batch, softmax=True)
            ce_loss = ce_loss_class(iout, label_batch)
            sum_dice_loss += dice_loss
            sum_ce_loss += ce_loss
            loss += (0.7 * dice_loss) + (0.3 * ce_loss)
    return sum_dice_loss.item(), sum_ce_loss.item(), loss.item()


class _StubModel(nn.Module):
    """태그로 고른 고정 텐서를 그대로 돌려준다.

    0*w 를 더해 grad 경로만 만든다 — 곱셈으로 엮으면 optimizer 가 w 를 갱신한 뒤 val forward 의
    출력이 달라져 기대값과 어긋난다.
    """

    def __init__(self, outs_by_tag):
        super().__init__()
        self.outs_by_tag = outs_by_tag
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        outs = self.outs_by_tag[int(x.flatten()[0].item())]
        zero = 0.0 * self.w.sum()
        if isinstance(outs, list):
            return [o + zero for o in outs]
        return outs + zero


class _StubLoader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class _StubWriter:
    def __init__(self, *args, **kwargs):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = value

    def add_image(self, *args, **kwargs):
        pass

    def close(self):
        pass


def run_trainer(outs_train, label_train, outs_val, label_val, strategy, dice_weight, ce_weight):
    """trainer_coca 를 stub 으로 1 epoch 돌려 tensorboard 스칼라와 로그를 회수한다."""
    model = _StubModel({TRAIN_TAG: outs_train, VAL_TAG: outs_val}).cuda()
    loaders = iter([
        _StubLoader([{'image': torch.full((1, 1, 1, 1), float(TRAIN_TAG)), 'label': label_train}]),
        _StubLoader([{'image': torch.full((1, 1, 1, 1), float(VAL_TAG)), 'label': label_val}]),
    ])
    writers = []

    args = types.SimpleNamespace(
        base_lr=1e-5, batch_size=B, img_size=HW, seed=1234, max_epochs=1,
        num_slices=1, fold_idx=0, hu_stats_path='<stub>',
        root_path_5fold='<stub>', list_dir_5fold='<stub>',
        supervision=strategy, dice_weight=dice_weight, ce_weight=ce_weight,
        early_stopping_patience=0, early_stopping_min_delta=0.0,
    )

    def make_writer(*a, **k):
        w = _StubWriter()
        writers.append(w)
        return w

    saved = {name: getattr(trainer_mod, name) for name in
             ('load_hu_stats', '_read_fold_list', 'COCAVolumeDataset', 'DataLoader',
              'SummaryWriter', 'tqdm')}
    root_handlers = logging.getLogger().handlers[:]
    captured = io.StringIO()
    try:
        trainer_mod.load_hu_stats = lambda path: None
        trainer_mod._read_fold_list = lambda list_dir, k: [f'case{k}']
        trainer_mod.COCAVolumeDataset = lambda *a, **k: [0]
        trainer_mod.DataLoader = lambda *a, **k: next(loaders)
        trainer_mod.SummaryWriter = make_writer
        trainer_mod.tqdm = lambda iterable, **k: iterable
        with tempfile.TemporaryDirectory() as snapshot_path:
            with contextlib.redirect_stdout(captured):
                trainer_mod.trainer_coca(args, model, snapshot_path)
    finally:
        for name, value in saved.items():
            setattr(trainer_mod, name, value)
        root = logging.getLogger()
        for handler in root.handlers[:]:
            if handler not in root_handlers:
                root.removeHandler(handler)
                handler.close()
    return writers[0].scalars, captured.getvalue()


def check(name, scalars, expected, prefix_pairs):
    exp_dice, exp_ce, exp_loss = expected
    for prefix, loss_tag in prefix_pairs:
        got = (scalars[f'{prefix}/dice_loss'], scalars[f'{prefix}/ce_loss'], scalars[loss_tag])
        assert got[0] == exp_dice, f'{name} {prefix}/dice_loss: {got[0]!r} != {exp_dice!r}'
        assert got[1] == exp_ce, f'{name} {prefix}/ce_loss: {got[1]!r} != {exp_ce!r}'
        assert got[2] == exp_loss, f'{name} {loss_tag}: {got[2]!r} != {exp_loss!r}'


torch.manual_seed(20240805)
label_train = torch.randint(0, C, (B, HW, HW)).cuda()
label_val = torch.randint(0, C, (B, HW, HW)).cuda()
single_train = torch.randn(B, C, HW, HW).cuda() * 3.0
single_val = torch.randn(B, C, HW, HW).cuda() * 3.0
multi_train = [torch.randn(B, C, HW, HW).cuda() * 3.0 for _ in range(4)]
multi_val = [torch.randn(B, C, HW, HW).cuda() * 3.0 for _ in range(4)]

# SMP 단일 출력 + last_layer + 0.5/0.5 → 통합 전 main 과 같아야 한다.
scalars, log_text = run_trainer(single_train, label_train, single_val, label_val,
                                'last_layer', 0.5, 0.5)
check('SMP', scalars, frozen_smp_loss(single_train, label_train), [('train', 'train/train_loss')])
check('SMP', scalars, frozen_smp_loss(single_val, label_val), [('val', 'val/val_loss')])

# ss 캐시로 supervision 로그는 trainer_coca 호출당 1회 (통합 전에는 train/val 각 1회). 손실과 무관.
assert log_text.count('Supervision strategy:') == 1, log_text

# EMCAD 4출력: 세 전략 모두 통합 전 EMCAD 와 같아야 한다. mutation 만 검사하면
# args.supervision 을 'mutation' 으로 하드코딩해도 통과해 전략 배선이 안 걸린다.
for strategy in ('mutation', 'deep_supervision', 'last_layer'):
    scalars, _ = run_trainer(multi_train, label_train, multi_val, label_val, strategy, 0.7, 0.3)
    check(f'EMCAD/{strategy}', scalars, frozen_emcad_loss(multi_train, label_train, strategy),
          [('train', 'train/train_loss')])
    check(f'EMCAD/{strategy}', scalars, frozen_emcad_loss(multi_val, label_val, strategy),
          [('val', 'val/val_loss')])

print('OK: trainer.py 손실 본문이 train/val 양쪽에서 통합 전 main(0.5/0.5)·'
      'EMCAD(0.7/0.3, mutation/deep_supervision/last_layer) 와 동일')
