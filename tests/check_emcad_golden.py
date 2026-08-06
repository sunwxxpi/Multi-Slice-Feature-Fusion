"""EMCADNet / EMCAD_SA_Net 골든 테스트: unify-branches 의 통합 코드가 frozen EMCAD/EMCAD-SA
브랜치와 동일한 파라미터 초기화·forward(eval/train)·gradient 를 재현하는지 검증한다.

기존 8개 테스트는 모듈 이름/개수/방향성 같은 구조적 속성만 확인해 `_fuse` 의 residual 제거나
prev/main/next 인터리빙 붕괴 같은 산술적 변경을 못 잡는다. frozen 소스는 저장소에 벤더링하지
않고 매 실행마다 git 커밋에서 직접 읽어와 스크래치 디렉터리에 격리 import 한다.

GPU 비교는 하지 않는다 (반드시 CPU 로 실행할 것): memory-efficient SDPA 와 bilinear-upsample
backward 가 GPU 에서는 같은 모델을 두 번 돌려도 자체 재현이 안 될 만큼 비결정적이라, GPU
gradient 비교는 구조적으로 flaky 하다.
"""
import contextlib
import importlib
import io
import os
import subprocess
import sys

import torch

# CPU 라도 멀티스레드 conv/attention backward 는 reduction 순서가 스레드 스케줄링에 따라 달라져
# 비결정적이다 — 같은 frozen 모델을 두 번 돌리는 self-consistency 조차 이 설정 없이는 실패했다
# (gradient 최대 0.008 오차, empirically 확인). 단일 스레드로 고정해야 CPU 에서 재현 가능하다.
torch.set_num_threads(1)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH_ROOT = ("/tmp/claude-1004/-home-psw-SAU-Net/797139ad-2f95-45db-a0f1-61976fc9b863/"
                "scratchpad/check_emcad_golden")
sys.path.insert(0, REPO)

FILES = ('resnet.py', 'decoders.py', 'pvtv2.py', 'networks.py')
CONSTRUCT_SEED = 12345
FORWARD_SEED = 999

# (label, frozen 브랜치, 클래스 이름, encoder, 입력 채널 수, unified 전용 추가 kwargs)
CASES = [
    ('EMCADNet', 'EMCAD', 'EMCADNet', 'pvt_v2_b1', 1, dict(use_msffm=False)),
    ('EMCAD_SA_Net', 'EMCAD-SA', 'EMCAD_SA_Net', 'pvt_v2_b1', 3, dict()),
]


def _git_show(ref, path):
    r = subprocess.run(['git', 'show', f'{ref}:{path}'], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"git show {ref}:{path} 실패 — frozen 레퍼런스 브랜치 '{ref}' 가 없어졌을 수 있음 "
            f"(현재 저장소의 브랜치 목록에 있는지 `git branch -a` 로 확인할 것). stderr={r.stderr}")
    return r.stdout


def materialize_frozen(ref):
    """ref 브랜치의 networks/emcad/*.py 를 스크래치 디렉터리에 그대로 옮겨 적는다 (저장소에는 안 씀)."""
    dest = os.path.join(SCRATCH_ROOT, f'frozen_{ref.replace("-", "_")}')
    pkg_dir = os.path.join(dest, 'networks', 'emcad')
    os.makedirs(pkg_dir, exist_ok=True)
    for fname in FILES:
        content = _git_show(ref, f'networks/emcad/{fname}')
        with open(os.path.join(pkg_dir, fname), 'w', encoding='utf-8') as f:
            f.write(content)
    return dest


def load_frozen_networks_module(dest_dir):
    """dest_dir 를 sys.path 맨 앞에 넣고 networks.emcad.networks 를 격리 import 한다.

    통합 브랜치의 networks.* 모듈과 이름이 겹치므로, import 직전에 캐시된 항목을 전부 빼뒀다가
    끝나고 그대로 복원한다 — 그래야 이 함수를 두 번 불러도(EMCAD, EMCAD-SA) 서로 안 섞이고,
    호출 전에 이미 import 돼 있던 통합 브랜치 모듈 객체도 그대로 살아남는다.
    """
    saved_mods = {k: sys.modules.pop(k) for k in list(sys.modules)
                 if k == 'networks' or k.startswith('networks.')}
    saved_path = sys.path[:]
    sys.path.insert(0, dest_dir)
    try:
        mod = importlib.import_module('networks.emcad.networks')
    finally:
        sys.path[:] = saved_path
        for k in list(sys.modules):
            if k == 'networks' or k.startswith('networks.'):
                del sys.modules[k]
        sys.modules.update(saved_mods)
    return mod


def _build(cls, encoder, seed, **extra):
    kwargs = dict(num_classes=5, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                 add=True, lgag_ks=3, activation='relu6', encoder=encoder, pretrain=False)
    kwargs.update(extra)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):  # 생성자의 param count print 억제
        return cls(**kwargs).to('cpu')


def _assert_params_match(frozen, unified, label):
    fp = dict(frozen.named_parameters())
    up = dict(unified.named_parameters())
    assert set(fp) == set(up), (
        f'{label}: 파라미터 이름 집합 불일치. frozen에만={set(fp)-set(up)} unified에만={set(up)-set(fp)}')
    mismatched = [n for n in fp if fp[n].shape != up[n].shape or not torch.equal(fp[n], up[n])]
    assert not mismatched, (
        f'{label}: 동일 seed 로 생성했는데 값이 다른 파라미터 {len(mismatched)}개: {mismatched[:5]}')
    return fp, up


def _forward(model, x, mode, seed):
    getattr(model, mode)()
    torch.manual_seed(seed)  # DropPath 가 RNG 를 쓰는 train 모드 비교를 위해 forward 직전에 시드 고정.
    return model(x)


def _assert_outputs_match(out_f, out_u, label):
    diffs = [(a - b).abs().max().item() for a, b in zip(out_f, out_u)]
    assert all(d == 0.0 for d in diffs), f'{label}: forward 출력 불일치 (head별 max abs diff)={diffs}'


def _assert_grads_match(frozen, unified, out_f, out_u, label):
    # 실제 라벨 없이 모든 파라미터로 그래디언트가 흐르게만 하면 되므로 출력 제곱합을 손실로 쓴다.
    sum(o.float().pow(2).sum() for o in out_f).backward()
    sum(o.float().pow(2).sum() for o in out_u).backward()
    fp = dict(frozen.named_parameters())
    up = dict(unified.named_parameters())
    bad = []
    for n in fp:
        gf, gu = fp[n].grad, up[n].grad
        if (gf is None) != (gu is None):
            bad.append((n, 'grad 존재 여부 불일치', gf is None, gu is None))
        elif gf is not None and not torch.equal(gf, gu):
            bad.append((n, (gf - gu).abs().max().item()))
    assert not bad, f'{label}: gradient 불일치 {len(bad)}개 파라미터: {bad[:5]}'
    frozen.zero_grad(set_to_none=True)
    unified.zero_grad(set_to_none=True)


def main():
    import networks.emcad.networks as unified_mod  # 통합 브랜치 코드 (검증 대상)

    for label, ref, cls_name, encoder, in_ch, extra_kwargs in CASES:
        dest = materialize_frozen(ref)
        frozen_mod = load_frozen_networks_module(dest)
        frozen_cls = getattr(frozen_mod, cls_name)
        unified_cls = getattr(unified_mod, cls_name)

        frozen = _build(frozen_cls, encoder, CONSTRUCT_SEED)
        unified = _build(unified_cls, encoder, CONSTRUCT_SEED, **extra_kwargs)
        _assert_params_match(frozen, unified, label)

        torch.manual_seed(0)
        x = torch.randn(1, in_ch, 64, 64)

        for mode in ('eval', 'train'):
            out_f = _forward(frozen, x, mode, FORWARD_SEED)
            out_u = _forward(unified, x, mode, FORWARD_SEED)
            _assert_outputs_match(out_f, out_u, f'{label}/{mode}')

        # gradient 비교는 DropPath 가 활성인 train() 상태에서 forward 를 다시 잡아 수행한다.
        out_f = _forward(frozen, x, 'train', FORWARD_SEED)
        out_u = _forward(unified, x, 'train', FORWARD_SEED)
        _assert_grads_match(frozen, unified, out_f, out_u, label)

        print(f'  - {label} (encoder={encoder}, frozen={ref}): 파라미터 {len(dict(frozen.named_parameters()))}개, '
             f'eval/train forward, gradient 모두 일치')


if __name__ == '__main__':
    main()
    print('OK: EMCADNet/EMCAD_SA_Net 이 frozen EMCAD/EMCAD-SA 와 파라미터·eval/train forward·gradient 모두 일치 (CPU)')
