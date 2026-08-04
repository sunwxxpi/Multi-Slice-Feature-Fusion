"""
5-fold CV 데이터셋 빌더 (1회용).

기존 COCA_3frames(train/test 각각 case0001~ 로 번호가 충돌)와 무관하게,
원본 nnUNet 포맷 Dataset001_COCA 에서 433 case 통합 풀을 새로 생성한다.

- case_id  : 원본 파일명의 전역 인덱스 그대로 (COCA_Tr_<gidx>_... -> case{gidx:04d})
             train 0~313, test 314~450 로 충돌 없이 433 unique.
- 저장 포맷: case 당 per-case 볼륨 .npy (memmap 효율). image=(D,H,W) f32, label=(D,H,W) u8.
- fold 분할: case 단위 (2.5D 인접 슬라이스 누수 방지), vessel multi-hot 층화,
             MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42).
- HU 정규화 상수: 433 case 전체 voxel 분포에서 1회 산출 (0.5% / 99.5% 분위수 + clip 후 mean/std).
- sample 네이밍: case{gidx:04d}_slice{n:03d}, n = triplet 시작 인덱스 0..D-3 (= D-2 개, 기존과 동일).
"""
import os
import csv
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

SRC = "/home/psw/SAU-Net/data/COCA/Dataset001_COCA"
OUT = "/home/psw/SAU-Net/data/COCA/COCA_3frames_5fold"
SPLITS = [
    ("train", "imagesTr", "labelsTr"),
    ("test", "imagesVal", "labelsVal"),
]
N_SPLITS = 5
SEED = 42
LOWER_PCT, UPPER_PCT = 0.5, 99.5
VESSELS = [1, 2, 3, 4]              # LCA, LAD, LCX, RCA
VNAMES = ["LCA", "LAD", "LCX", "RCA"]
HMIN, HMAX = -2000, 5000           # HU 히스토그램 정수 빈 범위 (여유 있게)


def global_index(basename):
    # "COCA_Tr_0_0001" -> 0,  "COCA_Val_314_0001" -> 314
    return int(basename.split("_")[2])


def main():
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "lists_COCA_5fold"), exist_ok=True)

    hist = np.zeros(HMAX - HMIN + 1, dtype=np.int64)
    cases = []

    for origin, imgdir, labdir in SPLITS:
        ipath = os.path.join(SRC, imgdir)
        lpath = os.path.join(SRC, labdir)
        for fn in tqdm(sorted(os.listdir(ipath)), desc=origin, unit="case"):
            if not fn.endswith("_0000.nii.gz"):
                continue
            base = fn.replace("_0000.nii.gz", "")
            gidx = global_index(base)
            cid = f"case{gidx:04d}"

            label_path = os.path.join(lpath, base + ".nii.gz")
            assert os.path.exists(label_path), f"label 없음: {label_path}"

            img = nib.load(os.path.join(ipath, fn)).get_fdata().astype(np.float32)  # (H,W,D)
            lab = nib.load(label_path).get_fdata().astype(np.uint8)                 # (H,W,D)
            assert img.shape == lab.shape, f"shape 불일치: {base}"
            D = img.shape[2]
            assert D >= 3, f"슬라이스 부족(D={D}): {base}"

            # (D,H,W) C-contiguous -> vol[c-1:c+2] 가 연속 평면이라 mmap 부분읽기에 최적
            img_dhw = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
            lab_dhw = np.ascontiguousarray(np.transpose(lab, (2, 0, 1)))
            np.save(os.path.join(OUT, "images", cid + ".npy"), img_dhw)
            np.save(os.path.join(OUT, "labels", cid + ".npy"), lab_dhw)

            clipped = np.clip(np.round(img_dhw), HMIN, HMAX).astype(np.int64)
            hist += np.bincount((clipped - HMIN).ravel(), minlength=hist.size)

            present = set(int(v) for v in np.unique(lab_dhw))
            multihot = [1 if v in present else 0 for v in VESSELS]
            cases.append(dict(cid=cid, origin=origin, src_base=base,
                              D=int(D), n_samples=int(D - 2), multihot=multihot))

    cases.sort(key=lambda c: c["cid"])
    cids = [c["cid"] for c in cases]
    assert len(cids) == len(set(cids)), "case_id 중복 발생"

    # ---- HU 정규화 상수 (히스토그램에서 산출) ----
    total = int(hist.sum())
    vals = np.arange(HMIN, HMAX + 1)
    cum = np.cumsum(hist)

    def pct(p):
        i = int(np.searchsorted(cum, p / 100.0 * total))
        return float(vals[min(i, len(vals) - 1)])

    lower, upper = pct(LOWER_PCT), pct(UPPER_PCT)
    cvals = np.clip(vals, lower, upper).astype(np.float64)
    mean = float((cvals * hist).sum() / total)
    var = float((hist * (cvals - mean) ** 2).sum() / total)
    std = float(np.sqrt(var))
    hu = dict(lower=lower, upper=upper, mean=mean, std=std,
              lower_pct=LOWER_PCT, upper_pct=UPPER_PCT,
              n_cases=len(cases), n_voxels=total)
    with open(os.path.join(OUT, "hu_stats_433.json"), "w") as f:
        json.dump(hu, f, indent=2)

    # ---- case 단위 stratified 5-fold ----
    X = np.arange(len(cases))
    Y = np.array([c["multihot"] for c in cases])
    mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_of = {}
    for fold, (_, te) in enumerate(mskf.split(X, Y)):
        for i in te:
            fold_of[cases[i]["cid"]] = fold

    # ---- fold 리스트 (case -> slice 펼치기) ----
    for k in range(N_SPLITS):
        lines = []
        for c in cases:
            if fold_of[c["cid"]] != k:
                continue
            lines += [f'{c["cid"]}_slice{n:03d}' for n in range(c["n_samples"])]
        with open(os.path.join(OUT, "lists_COCA_5fold", f"fold{k}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    with open(os.path.join(OUT, "lists_COCA_5fold", "fold_assignment.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "fold", "label_LCA", "label_LAD", "label_LCX", "label_RCA", "slice_count"])
        for c in cases:
            w.writerow([c["cid"], fold_of[c["cid"]], *c["multihot"], c["n_samples"]])

    with open(os.path.join(OUT, "case_index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "origin", "src_basename", "depth", "n_samples"])
        for c in cases:
            w.writerow([c["cid"], c["origin"], c["src_base"], c["D"], c["n_samples"]])

    # ---- 리포트 ----
    print("\n=== HU stats ===")
    print(json.dumps(hu, indent=2))
    print("\n=== fold balance (case 수) ===")
    print(f'{"":7}' + "".join(f"{'F'+str(k):>7}" for k in range(N_SPLITS)) + f'{"Total":>8}')
    for vi, vn in enumerate(VNAMES):
        row = [sum(1 for c in cases if fold_of[c["cid"]] == k and c["multihot"][vi]) for k in range(N_SPLITS)]
        print(f'{vn:7}' + "".join(f"{x:7}" for x in row) + f"{sum(row):8}")
    cs = [sum(1 for c in cases if fold_of[c["cid"]] == k) for k in range(N_SPLITS)]
    print(f'{"Cases":7}' + "".join(f"{x:7}" for x in cs) + f"{sum(cs):8}")
    ss = [sum(c["n_samples"] for c in cases if fold_of[c["cid"]] == k) for k in range(N_SPLITS)]
    print(f'{"Slices":7}' + "".join(f"{x:7}" for x in ss) + f"{sum(ss):8}")


if __name__ == "__main__":
    main()
