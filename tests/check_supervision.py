"""build_supervision 이 통합 전 두 계열의 손실 조합을 그대로 재현하는지 검증."""
from utils import build_supervision

# 단일 출력(SMP): 전략과 무관하게 최종단 하나. 통합 전 main 의 loss = 0.5*Dice + 0.5*CE 와 동일해진다.
for strategy in ('last_layer', 'mutation', 'deep_supervision'):
    assert build_supervision(strategy, 1) == [(-1,)], strategy

# EMCAD mutation: 4개 출력의 공집합 제외 부분집합 = 15개. 원본이 `if not s: continue` 로
# 걸렀던 공집합을 애초에 만들지 않는다.
ss = build_supervision('mutation', 4)
assert len(ss) == 15, len(ss)
assert () not in ss
assert (0,) in ss and (0, 1, 2, 3) in ss
assert len(set(ss)) == 15, '중복 조합 존재'
assert all(all(0 <= i < 4 for i in s) for s in ss), '범위 밖 인덱스'

# 조합 순서까지 통합 전 EMCAD 의 재귀 powerset 과 같아야 한다. 집합만 같고 순서가 다르면
# 15개 손실의 누적 순서가 바뀌어 fp32 최말단 비트(실측 1 ulp)가 통합 전과 달라진다.
assert ss == [(0, 1, 2, 3), (1, 2, 3), (0, 2, 3), (2, 3),
              (0, 1, 3), (1, 3), (0, 3), (3,),
              (0, 1, 2), (1, 2), (0, 2), (2,),
              (0, 1), (1,), (0,)], ss

# deep_supervision: 각 출력 단계 하나씩
assert build_supervision('deep_supervision', 4) == [(0,), (1,), (2,), (3,)]

# last_layer: 최종단만
assert build_supervision('last_layer', 4) == [(-1,)]

print('OK: build_supervision 이 단일 출력·mutation(15)·deep_supervision·last_layer 를 재현')
