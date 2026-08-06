"""build_3d_volume 이 채널 수와 무관하게 center 슬라이스를 집는지 검증.

3채널이면 index 1, 1채널이면 index 0 이 center 다. 채널 1 하드코딩은 1채널에서 IndexError.
"""
import types
import numpy as np

from tester import build_3d_volume

DEPTH, H, W = 4, 8, 8
args = types.SimpleNamespace(is_savenii=False, z_spacing=3)

def run(num_slices):
    image_slices, pred_slices, label_slices = {}, {}, {}
    for z in range(DEPTH):
        img = np.zeros((num_slices, H, W), dtype=np.uint8)
        # center 채널에만 z+1 을 채운다. 다른 채널은 0 이므로 잘못된 채널을 집으면 값이 0 이 된다.
        img[num_slices // 2] = z + 1
        image_slices[z] = img
        pred_slices[z] = np.full((H, W), z + 1, dtype=np.uint8)
        label_slices[z] = np.full((H, W), z + 1, dtype=np.uint8)
    return build_3d_volume(image_slices, pred_slices, label_slices, 'case0000', args, None)

for ns in (3, 1):
    image_3d, pred_3d, label_3d = run(ns)
    assert image_3d.shape == (DEPTH, H, W), (ns, image_3d.shape)
    for z in range(DEPTH):
        assert image_3d[z].max() == z + 1, \
            f'num_slices={ns}, z={z}: center 가 아닌 채널을 집었다 (got {image_3d[z].max()}, want {z + 1})'
    assert pred_3d.shape == label_3d.shape == (DEPTH, H, W)

print('OK: build_3d_volume 이 1채널·3채널 모두에서 center 슬라이스를 집는다')
