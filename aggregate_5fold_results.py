"""완료된 최종 학습의 5-fold 평가 결과와 지정 epoch 수를 집계한다."""
import os
import re
import argparse
import numpy as np
from training_protocol import load_final_training_record

VNAMES = ["LCA", "LAD", "LCX", "RCA"]   # class 1..4
NETCLASS = {"unet": "Unet", "segformer": "Segformer",
            "emcad": "EMCADNet", "emcad_sa": "EMCAD_SA_Net"}

NUM = r"(nan|[-+]?\d*\.?\d+)"
RE_CLASS = re.compile(rf"\[3D\] Class (\d+) - Dice: {NUM}, mIoU: {NUM}, HD: {NUM}")
RE_MEAN = re.compile(rf"\[3D\] Testing Performance - Mean Dice: {NUM}, Mean mIoU: {NUM}, Mean HD: {NUM}")
RE_CHECKPOINT = re.compile(r"Evaluation checkpoint: (.+)")


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parse_results(path):
    """마지막 최종 체크포인트 평가의 클래스별·평균 메트릭을 반환한다."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as fp:
        text = fp.read()
    checkpoints = list(RE_CHECKPOINT.finditer(text))
    if not checkpoints:
        return None
    checkpoint = checkpoints[-1].group(1).strip()
    text = text[checkpoints[-1].end():]
    cls = {}  # class_idx -> (dice, miou, hd) (마지막 등장값)
    for m in RE_CLASS.finditer(text):
        cls[int(m.group(1))] = (f(m.group(2)), f(m.group(3)), f(m.group(4)))
    means = RE_MEAN.findall(text)
    mean = means[-1] if means else None
    if set(cls) != {1, 2, 3, 4} or mean is None:
        return None
    out = {}
    out['checkpoint'] = checkpoint
    for mi, key in enumerate(("Dice", "mIoU", "HD")):
        row = [cls.get(c, (float("nan"),) * 3)[mi] for c in (1, 2, 3, 4)]
        row.append(f(mean[mi]) if mean else float("nan"))
        out[key] = row
    return out


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
    ap.add_argument("--epochs_per_fold", type=int, nargs=5,
                    help="fold 0부터 4까지 각각 최종 학습에 지정한 epoch 수")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--base_lr", type=float, default=0.00001)
    ap.add_argument("--test_log_root", default="./test_log")
    ap.add_argument("--model_root", default="./model")
    ap.add_argument("--results_dir", default="./results",
                    help="aggregate MD 출력 디렉터리 (gitignored)")
    ap.add_argument("--out", default=None,
                    help="명시 시 results_dir 무시")
    args = ap.parse_args()

    netcls = NETCLASS[args.decoder]
    sub = os.path.join(f"{netcls}_{args.encoder}", f"{args.dataset}_{args.img_size}")

    per_fold = {}
    ckpts = {}
    for k in range(5):
        epochs = args.epochs_per_fold[k] if args.epochs_per_fold else args.max_epochs
        param = f"epo{epochs}_bs{args.batch_size}_lr{args.base_lr}"
        exp = args.exp_template.format(fold=k)
        test_dir = os.path.join(args.test_log_root, sub, exp, param)
        model_dir = os.path.join(args.model_root, sub, exp, param)
        per_fold[k] = parse_results(os.path.join(test_dir, "results.txt"))
        try:
            ckpts[k] = load_final_training_record(model_dir, k, epochs)
        except (OSError, ValueError) as error:
            ckpts[k] = None
            per_fold[k] = None
            print(f"[warn] fold{k}: 최종 학습 기록 확인 실패: {error}")
        if per_fold[k] is not None:
            recorded = per_fold[k]['checkpoint']
            expected = os.path.join(model_dir, 'final_model.pth')
            if recorded is None or os.path.realpath(recorded) != os.path.realpath(expected):
                per_fold[k] = None
                print(f"[warn] fold{k}: 평가에 사용한 최종 체크포인트가 일치하지 않습니다.")
        if per_fold[k] is None:
            print(f"[warn] fold{k}: 유효한 최종 평가 결과 없음 -> {test_dir}/results.txt")

    title = args.exp_template.replace("_fold{fold}", "").replace("{fold}", "")
    parts = [f"# 5-Fold Results — {title}", ""]
    parts.append("## Run Summary")
    parts.append("")
    parts.append("| Fold | Selected Epochs | Completed Epochs | Checkpoint |")
    parts.append("|---|---|---|---|")
    for k in range(5):
        record = ckpts[k]
        if record is None:
            parts.append(f"| {k} | — | — | — |")
        else:
            parts.append(f"| {k} | {record['selected_epochs']} | {record['completed_epochs']} | final_model.pth |")
    parts.append("")
    parts.append(metric_table("Dice", per_fold, "Dice"))
    parts.append(metric_table("mIoU", per_fold, "mIoU"))
    parts.append(metric_table("HD (Surface Distance)", per_fold, "HD", nd=2))

    md = "\n".join(parts)
    print(md)
    out = args.out or os.path.join(args.results_dir, f"{title}.md")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fp:
        fp.write(md + "\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
