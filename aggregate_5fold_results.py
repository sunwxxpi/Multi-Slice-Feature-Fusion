"""
5-fold CV 결과 집계.

각 fold 의 test_log/.../results.txt 에서 클래스별 3D 메트릭(Dice/mIoU/HD)을 파싱하고,
model/.../*_best_model.pth 파일명에서 best epoch / best val_loss 를 읽어,
fold 별 값 + mean±std 를 Markdown 표로 출력한다.

예:
  python aggregate_5fold_results.py \
      --exp_template review_5fold_msffm_resnet50_unet_fold{fold}_seed42 \
      --encoder resnet50_sa --decoder unet
"""
import os
import re
import csv
import glob
import argparse
import numpy as np

VNAMES = ["LCA", "LAD", "LCX", "RCA"]   # class 1..4
NETCLASS = {"unet": "Unet", "segformer": "Segformer"}

NUM = r"(nan|[-+]?\d*\.?\d+)"
RE_CLASS = re.compile(rf"\[3D\] Class (\d+) - Dice: {NUM}, mIoU: {NUM}, HD: {NUM}")
RE_MEAN = re.compile(rf"\[3D\] Testing Performance - Mean Dice: {NUM}, Mean mIoU: {NUM}, Mean HD: {NUM}")
RE_BEST = re.compile(r"epoch_(\d+)_([\d.]+)_best_model\.pth$")
RE_STOP = re.compile(r"Early stopping at epoch (\d+)")


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parse_results(path):
    """results.txt 에서 마지막 실행의 클래스별/평균 메트릭을 파싱.
    반환: dict metric -> [LCA,LAD,LCX,RCA, Mean] (값 없으면 None)."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as fp:
        text = fp.read()
    cls = {}  # class_idx -> (dice, miou, hd) (마지막 등장값)
    for m in RE_CLASS.finditer(text):
        cls[int(m.group(1))] = (f(m.group(2)), f(m.group(3)), f(m.group(4)))
    means = RE_MEAN.findall(text)
    mean = means[-1] if means else None
    if not cls:
        return None
    out = {}
    for mi, key in enumerate(("Dice", "mIoU", "HD")):
        row = [cls.get(c, (float("nan"),) * 3)[mi] for c in (1, 2, 3, 4)]
        row.append(f(mean[mi]) if mean else float("nan"))
        out[key] = row
    return out


def parse_best_ckpt(model_dir):
    """*_best_model.pth 파일명에서 (best_epoch, best_val_loss) 추출."""
    cks = glob.glob(os.path.join(model_dir, "*_best_model.pth"))
    if not cks:
        return (None, None)
    m = RE_BEST.search(os.path.basename(cks[0]))
    return (int(m.group(1)), float(m.group(2))) if m else (None, None)


def parse_stop_epoch(model_dir):
    """train log.txt 에서 early stopping / 마지막 epoch 추출 (best-effort)."""
    log = os.path.join(model_dir, "log.txt")
    if not os.path.exists(log):
        return None
    with open(log, "r") as fp:
        text = fp.read()
    s = RE_STOP.findall(text)
    if s:
        return int(s[-1])
    ep = re.findall(r"Train - epoch (\d+)", text)
    return int(ep[-1]) if ep else None


def fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def metric_table(title, per_fold, key, nd=4):
    lines = [f"## {title}", "",
             "| Fold | " + " | ".join(VNAMES) + " | Mean |",
             "|" + "---|" * 6]
    mat = []
    for k in range(5):
        r = per_fold[k][key] if per_fold[k] else [float("nan")] * 5
        mat.append(r)
        lines.append(f"| {k} | " + " | ".join(fmt(v, nd) for v in r) + " |")
    mat = np.array(mat, dtype=float)
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    cells = [f"{fmt(mean[i], nd)} ± {fmt(std[i], nd)}" for i in range(5)]
    lines.append("| **mean ± std** | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_template", required=True,
                    help="fold 자리에 {fold} 를 둔 exp_setting 템플릿")
    ap.add_argument("--encoder", default="resnet50_sa")
    ap.add_argument("--decoder", default="unet", choices=list(NETCLASS))
    ap.add_argument("--dataset", default="COCA")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--max_epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--base_lr", type=float, default=0.00001)
    ap.add_argument("--test_log_root", default="./test_log")
    ap.add_argument("--model_root", default="./model")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    netcls = NETCLASS[args.decoder]
    param = f"epo{args.max_epochs}_bs{args.batch_size}_lr{args.base_lr}"
    sub = os.path.join(f"{netcls}_{args.encoder}", f"{args.dataset}_{args.img_size}")

    per_fold = {}
    ckpts = {}
    for k in range(5):
        exp = args.exp_template.format(fold=k)
        test_dir = os.path.join(args.test_log_root, sub, exp, param)
        model_dir = os.path.join(args.model_root, sub, exp, param)
        per_fold[k] = parse_results(os.path.join(test_dir, "results.txt"))
        be, bv = parse_best_ckpt(model_dir)
        ckpts[k] = (be, bv, parse_stop_epoch(model_dir))
        if per_fold[k] is None:
            print(f"[warn] fold{k}: results 없음 -> {test_dir}/results.txt")

    title = args.exp_template.replace("_fold{fold}", "").replace("{fold}", "")
    parts = [f"# 5-Fold Results — {title}", ""]
    parts.append("## Run Summary")
    parts.append("")
    parts.append("| Fold | Best Epoch | Best Val Loss | Stop Epoch |")
    parts.append("|---|---|---|---|")
    for k in range(5):
        be, bv, se = ckpts[k]
        parts.append(f"| {k} | {be if be is not None else '—'} | "
                     f"{fmt(bv) if bv is not None else '—'} | {se if se is not None else '—'} |")
    parts.append("")
    parts.append(metric_table("Dice", per_fold, "Dice"))
    parts.append(metric_table("mIoU", per_fold, "mIoU"))
    parts.append(metric_table("HD (Surface Distance)", per_fold, "HD", nd=2))

    md = "\n".join(parts)
    print(md)
    out = args.out or f"aggregate_5fold_{title}.md"
    with open(out, "w") as fp:
        fp.write(md + "\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
