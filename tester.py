import logging
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.metrics import precision_recall_curve, auc
from scipy.ndimage import zoom
from scipy.spatial.distance import directed_hausdorff
from datasets.dataset import COCA_dataset, ToTensor

def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute Soerensen-Dice coefficient."""
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum

def compute_average_precision(mask_gt, mask_pred):
    """Compute Average Precision (AP) score."""
    precision, recall, _ = precision_recall_curve(mask_gt.flatten(), mask_pred.flatten())
    return auc(recall, precision)

def compute_hausdorff_distance(mask_gt, mask_pred):
    """Compute Hausdorff Distance (HD)."""
    gt_points = np.transpose(np.nonzero(mask_gt))
    pred_points = np.transpose(np.nonzero(mask_pred))
    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.NaN
    hd_1 = directed_hausdorff(gt_points, pred_points)[0]
    hd_2 = directed_hausdorff(pred_points, gt_points)[0]
    return max(hd_1, hd_2)

def calculate_metric_percase(pred, gt):
    """
    pred, gt가 이미 특정 클래스에 대한 mask로 binary 형태라고 가정.
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if gt.sum() == 0 and pred.sum() == 0:
        dice = 1
        m_ap = 1
        hd = 0
    elif gt.sum() == 0 and pred.sum() > 0:
        dice = 0
        m_ap = 0
        hd = np.NaN
    else:
        dice = compute_dice_coefficient(gt, pred)
        m_ap = compute_average_precision(gt, pred)
        hd = compute_hausdorff_distance(gt, pred)

    return dice, m_ap, hd

def test_single_volume(image, label, model, classes, patch_size=[512, 512], test_save_path=None, case=None, z_spacing=1):
    """
    한 볼륨(3D)을 받아서, 2D slice 단위로 추론 & 3D 예측(prediction_3d) 구성 후
    클래스별 3D 메트릭을 계산해 반환.
    또한, 2D 슬라이스 중 병변이 실제로 존재하는 슬라이스에 대해서만
    Dice/mAP를 계산해 별도 리스트로 저장해둠(2D LesionOnly).
    """
    image_np = image.squeeze().cpu().detach().numpy()  # (D, H, W)
    label_np = label.squeeze().cpu().detach().numpy()  # (D, H, W)

    D, H, W = image_np.shape
    prediction_3d = np.zeros_like(label_np, dtype=np.uint8)

    # (2D LesionOnly) 각 클래스별로 (Dice_2d, mAP_2d) 저장
    lesion_slice_metrics_2d = {c: [] for c in range(1, classes)}

    for d in range(D):
        slice_2d = image_np[d, ...]
        label_2d = label_np[d, ...]

        x, y = slice_2d.shape
        if (x != patch_size[0]) or (y != patch_size[1]):
            slice_2d = zoom(slice_2d, (patch_size[0]/x, patch_size[1]/y), order=3)

        input_tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).float().cuda()
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            out_2d = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()
        
        if (x != patch_size[0]) or (y != patch_size[1]):
            out_2d = zoom(out_2d, (x/patch_size[0], y/patch_size[1]), order=0)

        prediction_3d[d, ...] = out_2d

        # 2D LesionOnly 메트릭: 실제 병변이 있는 슬라이스만
        for c in range(1, classes):
            gt_mask_2d = (label_2d == c)
            if gt_mask_2d.sum() > 0:
                pred_mask_2d = (out_2d == c)
                dice_2d, map_2d, _ = calculate_metric_percase(pred_mask_2d, gt_mask_2d)
                lesion_slice_metrics_2d[c].append((dice_2d, map_2d))

    # 3D 볼륨 메트릭
    metric_list_3d = []
    for c in range(1, classes):
        dice, m_ap, hd = calculate_metric_percase(prediction_3d == c, label_np == c)
        metric_list_3d.append((dice, m_ap, hd))

    # 예측 결과 저장
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image_np.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction_3d.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label_np.astype(np.float32))

        img_itk.SetSpacing((0.375, 0.375, z_spacing))
        prd_itk.SetSpacing((0.375, 0.375, z_spacing))
        lab_itk.SetSpacing((0.375, 0.375, z_spacing))

        sitk.WriteImage(prd_itk, f"{test_save_path}/{case}_pred.nii.gz")
        sitk.WriteImage(img_itk, f"{test_save_path}/{case}_img.nii.gz")
        sitk.WriteImage(lab_itk, f"{test_save_path}/{case}_gt.nii.gz")

    return metric_list_3d, lesion_slice_metrics_2d

def inference(args, model, test_save_path=None):
    test_transform = T.Compose([
        ToTensor()
    ])
    db_test = COCA_dataset(
        base_dir=args.volume_path, 
        split="test", 
        list_dir=args.list_dir,
        transform=test_transform
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    logging.info(f"{len(testloader)} test iterations per epoch")
    model.eval()
    
    metric_list_all_3d = []
    lesion_slice_metrics_allcases_2d = {c: [] for c in range(1, args.num_classes)}

    for i_batch, sampled_batch in enumerate(testloader, start=1):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        
        metric_3d, lesion_2d = test_single_volume(
                                    image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                    test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing
                                    )

        metric_list_all_3d.append(metric_3d)

        # 2D LesionOnly 누적
        for c in range(1, args.num_classes):
            lesion_slice_metrics_allcases_2d[c].extend(lesion_2d[c])

        # 케이스별 3D 평균
        mean_dice_case = np.nanmean(metric_3d, axis=0)[0]
        mean_m_ap_case = np.nanmean(metric_3d, axis=0)[1]
        mean_hd95_case = np.nanmean(metric_3d, axis=0)[2]
        logging.info(f"{case_name} - Dice: {mean_dice_case:.4f}, mAP: {mean_m_ap_case:.4f}, HD95: {mean_hd95_case:.2f}")

    # (3D) 전체 요약
    metric_array = np.array(metric_list_all_3d)
    for i in range(1, args.num_classes):
        class_dice = np.nanmean(metric_array[:, i-1, 0])
        class_m_ap = np.nanmean(metric_array[:, i-1, 1])
        class_hd95 = np.nanmean(metric_array[:, i-1, 2])
        logging.info(f"[3D] Class {i} - Dice: {class_dice:.4f}, mAP: {class_m_ap:.4f}, HD95: {class_hd95:.2f}")

    mean_dice_3d = np.nanmean(metric_array[:,:,0])
    mean_m_ap_3d = np.nanmean(metric_array[:,:,1])
    mean_hd95_3d = np.nanmean(metric_array[:,:,2])
    logging.info(f"[3D] Testing performance - Mean Dice: {mean_dice_3d:.4f}, Mean mAP: {mean_m_ap_3d:.4f}, Mean HD95: {mean_hd95_3d:.2f}")

    # (2D LesionOnly) 병변 존재 슬라이스 결과
    all_classes_dice = []
    all_classes_map  = []
    for c in range(1, args.num_classes):
        slice_metrics_c = np.array(lesion_slice_metrics_allcases_2d[c])
        if len(slice_metrics_c) == 0:
            logging.info(f"[2D, LesionOnly] Class {c}: no lesion slice found.")
            continue

        dice_mean_c = np.nanmean(slice_metrics_c[:, 0])
        map_mean_c  = np.nanmean(slice_metrics_c[:, 1])
        all_classes_dice.append(dice_mean_c)
        all_classes_map.append(map_mean_c)
        logging.info(f"[2D, LesionOnly] (#slices: {len(slice_metrics_c)}) Class {c} - Dice: {dice_mean_c:.4f}, mAP: {map_mean_c:.4f}")

    if len(all_classes_dice) > 0:
        mean_dice_2d = np.mean(all_classes_dice)
        mean_map_2d  = np.mean(all_classes_map)
        logging.info(f"[2D, LesionOnly] Testing performance - Mean Dice: {mean_dice_2d:.4f}, Mean mAP: {mean_map_2d:.4f}")
    else:
        logging.info("[2D, LesionOnly] No lesion slices found for any class.")

    return "Testing Finished!"