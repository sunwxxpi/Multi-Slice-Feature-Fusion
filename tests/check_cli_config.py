"""CLI 설정 헬퍼 검증 — decoder별 허용 encoder 와 입력 슬라이스 수 유도.

train.py 와 test.py 가 같은 함수를 쓰게 해서 둘이 어긋나는 걸 구조적으로 막는다.
"""
import subprocess
import sys

from utils import (SMP_ENCODERS, EMCAD_ENCODERS, EMCAD_SA_ENCODERS,
                   allowed_encoders, derive_num_slices)

# 입력 슬라이스 수는 모델 구성이 결정한다: `_sa` encoder 는 3, 나머지는 1.
assert derive_num_slices('unet', 'resnet50_sa') == 3
assert derive_num_slices('unet', 'resnet50') == 1
assert derive_num_slices('segformer', 'mit_b2_sa') == 3
assert derive_num_slices('segformer', 'mit_b2') == 1

# SMP 목록은 plain 4개 + _sa 4개
assert len(SMP_ENCODERS) == 8, SMP_ENCODERS
for base in ('resnet50', 'densenet201', 'efficientnet-b4', 'mit_b2'):
    assert base in SMP_ENCODERS
    assert f'{base}_sa' in SMP_ENCODERS

# unet/segformer 는 SMP 목록 전체를 받는다
for decoder in ('unet', 'segformer'):
    assert allowed_encoders(decoder) == SMP_ENCODERS, decoder

# --- EMCAD 계열 ---

# emcad 는 1채널, emcad_sa 는 3채널
assert derive_num_slices('emcad', 'pvt_v2_b2') == 1
assert derive_num_slices('emcad_sa', 'pvt_v2_b2') == 3

# emcad_sa 는 pvt_v2_b1~b5 만 (NonLocalBlock 채널이 320/512 고정, b0 만 160/256)
assert EMCAD_SA_ENCODERS == ['pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5']
assert 'pvt_v2_b0' not in EMCAD_SA_ENCODERS
assert 'pvt_v2_b0' in EMCAD_ENCODERS

assert allowed_encoders('emcad') == EMCAD_ENCODERS
assert allowed_encoders('emcad_sa') == EMCAD_SA_ENCODERS

# 'resnet50' 은 SMP 와 EMCAD 양쪽에 있다. NetClass 가 다르므로 경로는 안 겹치지만
# decoder 별 허용 목록은 각자의 것을 써야 한다.
assert 'resnet50' in SMP_ENCODERS and 'resnet50' in EMCAD_ENCODERS

# --- 진입점 배선: train.py 와 test.py 가 실제로 이 헬퍼를 쓰는가 ---

PY = sys.executable

INVALID = [
    ('emcad_sa', 'pvt_v2_b0'),      # NonLocalBlock 채널 불일치
    ('emcad_sa', 'resnet50'),       # MSFFM 미지원 백본
    ('unet', 'pvt_v2_b2'),          # SMP 레지스트리에 없음
    ('segformer', 'pvt_v2_b2'),
    ('emcad', 'resnet50_sa'),       # SMP 전용 이름
]
VALID = [
    ('unet', 'resnet50'), ('unet', 'resnet50_sa'),
    ('segformer', 'mit_b2'), ('segformer', 'mit_b2_sa'),
    ('emcad', 'pvt_v2_b2'), ('emcad_sa', 'pvt_v2_b2'),
]

REJECT = '지원하지 않음'

def run(entry, decoder, encoder):
    # 존재하지 않는 hu_stats 경로를 준다. 조합 검증을 통과한 유효 조합은 그 직후
    # load_hu_stats(train) / 체크포인트 glob(test) 에서 죽는다 — 학습은 시작되지 않는다.
    # 이 인자를 빼면 실제 5-fold 학습(16,685 슬라이스)이 돌아간다.
    return subprocess.run(
        [PY, entry, '--decoder', decoder, '--encoder', encoder,
         '--exp_setting', '__validation_probe___fold0', '--max_epochs', '1', '--use_5fold_cv',
         '--hu_stats_path', '/nonexistent/__probe__.json'],
        capture_output=True, text=True, timeout=600)

failures = []
for entry in ('train.py', 'test.py'):
    for decoder, encoder in INVALID:
        r = run(entry, decoder, encoder)
        if REJECT not in (r.stderr + r.stdout):
            failures.append(f'{entry} {decoder}+{encoder}: 거부되지 않음 (exit={r.returncode})')
    for decoder, encoder in VALID:
        r = run(entry, decoder, encoder)
        out = r.stderr + r.stdout
        if REJECT in out:
            failures.append(f'{entry} {decoder}+{encoder}: 유효한 조합인데 거부됨')
        # 양성 증거: 검증을 실제로 통과해 다음 단계까지 갔는가.
        # 이게 없으면 import 에러 같은 무관한 실패도 전부 "통과" 로 집계된다.
        elif not any(k in out for k in ('__probe__.json', 'FileNotFoundError', 'IndexError')):
            failures.append(f'{entry} {decoder}+{encoder}: 검증 이후 단계 도달 증거 없음 (exit={r.returncode})')

if failures:
    raise AssertionError('진입점 조합 검증 실패:\n  ' + '\n  '.join(failures))

print(f'OK: 헬퍼 단위 검증 + 진입점 2개 x 무효 {len(INVALID)}개 거부, 유효 {len(VALID)}개 통과')
