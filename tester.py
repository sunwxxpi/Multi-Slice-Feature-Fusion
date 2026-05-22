import os
import logging
import math
import numpy as np
import SimpleITK as sitk
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms as T
from scipy.ndimage import zoom
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric, SurfaceDistanceMetric
from datasets.dataset import COCA_dataset, COCAVolumeDataset, load_hu_stats, Resize, ToTensor

def process_slice(slice_2d, model, patch_size):
    x, y = slice_2d.shape
    if (x, y) != tuple(patch_size):
        slice_2d = zoom(slice_2d, (patch_size[0] / x, patch_size[1] / y), order=3)

    input_tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).float().cuda()

    model.eval()
    with torch.no_grad():
        with autocast():
            outputs = model(input_tensor)
            out_2d = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()

    if (x, y) != tuple(patch_size):
        out_2d = zoom(out_2d, (x / patch_size[0], y / patch_size[1]), order=0)

    return out_2d

def test_single_volume(image, label, model, classes, patch_size, dice_metric, miou_metric, hd_metric, test_save_path=None, case=None, z_spacing=1):
    image_np, label_np = image.squeeze().cpu().detach().numpy(), label.squeeze().cpu().detach().numpy()
    D, H, W = image_np.shape
    max_hd_3d = math.sqrt((D-1)**2 + (H-1)**2 + (W-1)**2)
    prediction_3d = np.zeros_like(label_np, dtype=np.uint8)

    for d in range(D):
        slice_2d = image_np[d]
        out_2d = process_slice(slice_2d, model, patch_size)
        prediction_3d[d] = out_2d

    metrics_per_class = []

    logging.info(f"Metrics for Case: {case}")
    for c in range(1, classes):
        pred_class = (prediction_3d == c).astype(np.uint8)
        label_class = (label_np == c).astype(np.uint8)

        pred_tensor = torch.from_numpy(pred_class).unsqueeze(0).unsqueeze(0).float().cuda()
        label_tensor = torch.from_numpy(label_class).unsqueeze(0).unsqueeze(0).float().cuda()

        dice = dice_metric(pred_tensor, label_tensor)
        miou = miou_metric(pred_tensor, label_tensor)
        hd = hd_metric(pred_tensor, label_tensor)
        
        if isinstance(hd, torch.Tensor):
            hd = hd.item()

        gt_sum = label_class.sum()
        pred_sum = pred_class.sum()

        if gt_sum == 0 and pred_sum == 0:
            status = "(GT==0 & Pred==0)"
        elif gt_sum == 0 and pred_sum > 0:
            hd = np.nan
            status = "(GT==0 & Pred>0)"
        elif gt_sum > 0 and pred_sum == 0:
            hd = max_hd_3d
            status = "(GT>0 & Pred==0)"
        else:
            status = "(GT>0 & Pred>0)"

        logging.info(f"  Class {c}: Dice: {dice.item():.4f}, mIoU: {miou.item():.4f}, HD: {hd:.2f} {status}")
        
        metrics_per_class.append((dice.item(), miou.item(), hd))

    if test_save_path and case:
        for array, suffix in zip([image_np, prediction_3d, label_np], ["img", "pred", "gt"]):
            array = np.flip(np.transpose(array, (0, 2, 1)), (1, 2))
            itk_img = sitk.GetImageFromArray(array.astype(np.float32))
            itk_img.SetSpacing((0.375, 0.375, z_spacing))
            sitk.WriteImage(itk_img, f"{test_save_path}/{case}_{suffix}.nii.gz")

    return metrics_per_class

def inference(args, model, test_save_path=None):
    if getattr(args, 'use_5fold_cv', False):
        return inference_5fold(args, model, test_save_path)

    test_transform = T.Compose([ToTensor()])
    db_test = COCA_dataset(base_dir=args.volume_path,
                           list_dir=args.list_dir,
                           split="test",
                           transform=test_transform)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"{len(testloader)} test iterations per epoch")

    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=True)
    miou_metric = MeanIoU(include_background=True, reduction="mean", ignore_empty=True)
    # hd_metric = HausdorffDistanceMetric(include_background=True, distance_metric="euclidean", percentile=95)
    hd_metric = SurfaceDistanceMetric(include_background=False, symmetric=True, distance_metric="euclidean")

    metrics_3d_all = []

    for sampled_batch in testloader:
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metrics_3d = test_single_volume(image, label, model, 
                                        args.num_classes, [args.img_size, args.img_size], 
                                        dice_metric, miou_metric, hd_metric,
                                        test_save_path, case_name, args.z_spacing)

        metrics_3d_all.append(metrics_3d)

        mean_dice = np.nanmean([m[0] for m in metrics_3d])
        mean_miou = np.nanmean([m[1] for m in metrics_3d])
        mean_hd = np.nanmean([m[2] for m in metrics_3d])

        logging.info(f"  [Case {case_name}] - Mean Dice: {mean_dice:.4f}, Mean mIoU: {mean_miou:.4f}, Mean HD: {mean_hd:.2f}\n")

    metrics_3d_array = np.array(metrics_3d_all)

    logging.info("\nOverall 3D Metrics Across All Cases:")

    class_dice_means = []
    class_miou_means = []
    class_hd_means = []

    for c_idx in range(1, args.num_classes):
        dice_c = np.nanmean(metrics_3d_array[:, c_idx-1, 0])
        miou_c = np.nanmean(metrics_3d_array[:, c_idx-1, 1])
        hd_c = np.nanmean(metrics_3d_array[:, c_idx-1, 2])

        logging.info(f"  [3D] Class {c_idx} - Dice: {dice_c:.4f}, mIoU: {miou_c:.4f}, HD: {hd_c:.2f}")

        class_dice_means.append(dice_c)
        class_miou_means.append(miou_c)
        class_hd_means.append(hd_c)

    mean_dice_all = np.nanmean(class_dice_means)
    mean_miou_all = np.nanmean(class_miou_means)
    mean_hd_all = np.nanmean(class_hd_means)

    logging.info(f"  [3D] Testing Performance - Mean Dice: {mean_dice_all:.4f}, Mean mIoU: {mean_miou_all:.4f}, Mean HD: {mean_hd_all:.2f}")

    return "Testing Finished!"

def parse_case_and_slice_id(full_name):
    if '_slice' in full_name:
        case_id, slice_str = full_name.split('_slice', 1)
        return case_id, int(slice_str)
    return full_name, 0

def inference_5fold(args, model, test_save_path=None):
    """5-fold CV 평가 (MSFFM 미적용 baseline).

    학습 때의 validation fold(fold_idx)를 center 슬라이스 단위로 예측한 뒤 case 별
    3D 볼륨으로 합성해 MONAI 3D 메트릭을 계산한다. MSFFM 브랜치와 동일한 center
    슬라이스 집합·라벨·HU 정규화·메트릭 설정(include_background=True)을 써서 공정
    비교가 보장된다 (입력만 1채널). 레거시 볼륨 경로의 include_background=False 와
    달리, 비교 대상인 MSFFM 5-fold 평가와 메트릭을 일치시킨다.
    """
    test_transform = T.Compose([Resize(output_size=[args.img_size, args.img_size]),
                                ToTensor()])
    hu = load_hu_stats(args.hu_stats_path)
    with open(os.path.join(args.list_dir_5fold, f"fold{args.fold_idx}.txt"), 'r') as f:
        test_samples = [ln.strip() for ln in f if ln.strip()]
    db_test = COCAVolumeDataset(os.path.join(args.root_path_5fold, 'images'),
                                os.path.join(args.root_path_5fold, 'labels'),
                                test_samples, transform=test_transform, hu_stats=hu)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"{len(testloader)} test iterations per epoch")

    # case 별 {slice_id: 2D pred / label} 누적 (center 슬라이스만)
    pred_slices = {}
    label_slices = {}

    model.eval()
    for sampled_batch in testloader:
        image, label = sampled_batch["image"], sampled_batch["label"]
        case_name = sampled_batch['case_name'][0]
        case_id, slice_id = parse_case_and_slice_id(case_name)

        with torch.no_grad():
            with autocast():
                outputs = model(image.float().cuda())
                pred_2d = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        label_2d = label.squeeze(0).cpu().numpy().astype(np.uint8)

        if case_id not in pred_slices:
            pred_slices[case_id] = {}
            label_slices[case_id] = {}
        pred_slices[case_id][slice_id] = pred_2d
        label_slices[case_id][slice_id] = label_2d

    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=True)
    miou_metric = MeanIoU(include_background=True, reduction="mean", ignore_empty=True)
    hd_metric = SurfaceDistanceMetric(include_background=True, symmetric=True, distance_metric="euclidean")

    metrics_3d_all = []
    for case_id in sorted(pred_slices.keys()):
        sorted_ids = sorted(pred_slices[case_id].keys())
        pred_3d = np.stack([pred_slices[case_id][z] for z in sorted_ids], axis=0)
        label_3d = np.stack([label_slices[case_id][z] for z in sorted_ids], axis=0)

        D, H, W = pred_3d.shape
        max_hd_3d = math.sqrt((D-1)**2 + (H-1)**2 + (W-1)**2)

        metrics_per_class = []
        logging.info(f"Metrics for Case: {case_id}")
        for c in range(1, args.num_classes):
            pred_class = (pred_3d == c).astype(np.uint8)
            label_class = (label_3d == c).astype(np.uint8)

            pred_tensor = torch.from_numpy(pred_class).unsqueeze(0).unsqueeze(0).float().cuda()
            label_tensor = torch.from_numpy(label_class).unsqueeze(0).unsqueeze(0).float().cuda()

            dice = dice_metric(pred_tensor, label_tensor)
            miou = miou_metric(pred_tensor, label_tensor)
            hd = hd_metric(pred_tensor, label_tensor)
            if isinstance(hd, torch.Tensor):
                hd = hd.item()

            gt_sum = label_class.sum()
            pred_sum = pred_class.sum()
            if gt_sum == 0 and pred_sum == 0:
                status = "(GT==0 & Pred==0)"
            elif gt_sum == 0 and pred_sum > 0:
                hd = np.nan
                status = "(GT==0 & Pred>0)"
            elif gt_sum > 0 and pred_sum == 0:
                hd = max_hd_3d
                status = "(GT>0 & Pred==0)"
            else:
                status = "(GT>0 & Pred>0)"

            logging.info(f"  Class {c}: Dice: {dice.item():.4f}, mIoU: {miou.item():.4f}, HD: {hd:.2f} {status}")
            metrics_per_class.append((dice.item(), miou.item(), hd))

        metrics_per_class = np.array(metrics_per_class)
        logging.info(f"  [Case {case_id}] - Mean Dice: {np.nanmean(metrics_per_class[:, 0]):.4f}, "
                     f"Mean mIoU: {np.nanmean(metrics_per_class[:, 1]):.4f}, "
                     f"Mean HD: {np.nanmean(metrics_per_class[:, 2]):.2f}\n")
        metrics_3d_all.append(metrics_per_class)

    metrics_3d_array = np.array(metrics_3d_all)

    logging.info("\nOverall 3D Metrics Across All Cases:")
    class_dice_means, class_miou_means, class_hd_means = [], [], []
    for c_idx in range(1, args.num_classes):
        dice_c = np.nanmean(metrics_3d_array[:, c_idx-1, 0])
        miou_c = np.nanmean(metrics_3d_array[:, c_idx-1, 1])
        hd_c = np.nanmean(metrics_3d_array[:, c_idx-1, 2])
        logging.info(f"  [3D] Class {c_idx} - Dice: {dice_c:.4f}, mIoU: {miou_c:.4f}, HD: {hd_c:.2f}")
        class_dice_means.append(dice_c)
        class_miou_means.append(miou_c)
        class_hd_means.append(hd_c)

    logging.info(f"  [3D] Testing Performance - Mean Dice: {np.nanmean(class_dice_means):.4f}, "
                 f"Mean mIoU: {np.nanmean(class_miou_means):.4f}, Mean HD: {np.nanmean(class_hd_means):.2f}")

    return "Testing Finished!"