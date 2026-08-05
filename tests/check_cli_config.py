"""CLI 설정 헬퍼 검증 — decoder별 허용 encoder 와 입력 슬라이스 수 유도.

train.py 와 test.py 가 같은 함수를 쓰게 해서 둘이 어긋나는 걸 구조적으로 막는다.
"""
from utils import SMP_ENCODERS, allowed_encoders, derive_num_slices

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

print(f'OK: derive_num_slices/allowed_encoders (SMP {len(SMP_ENCODERS)}개)')
