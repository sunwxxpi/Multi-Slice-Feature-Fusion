import logging
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.metrics import precision_recall_curve, auc
from scipy.spatial.distance import directed_hausdorff
from datasets.dataset import COCA_dataset, Resize, ToTensor

# Dice coefficient 계산 함수
def compute_dice_coefficient(mask_gt, mask_pred):
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum

# Average Precision (AP) 계산 함수
def compute_average_precision(mask_gt, mask_pred):
    precision, recall, _ = precision_recall_curve(mask_gt.flatten(), mask_pred.flatten())
    return auc(recall, precision)

# Hausdorff Distance 계산 함수
def compute_hausdorff_distance(mask_gt, mask_pred):
    gt_points = np.transpose(np.nonzero(mask_gt))
    pred_points = np.transpose(np.nonzero(mask_pred))
    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.NaN
    hd_1 = directed_hausdorff(gt_points, pred_points)[0]
    hd_2 = directed_hausdorff(pred_points, gt_points)[0]
    return max(hd_1, hd_2)

# 각 사례(case)별 메트릭 계산 함수
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    # GT와 예측 모두 병변이 없을 경우
    if gt.sum() == 0 and pred.sum() == 0:
        dice = 1
        m_ap = 1
        hd = 0
    # GT는 병변이 없고 예측에만 병변이 있을 경우
    elif gt.sum() == 0 and pred.sum() > 0:
        dice = 0
        m_ap = 0
        hd = np.NaN
    # 일반적인 경우
    else:
        dice = compute_dice_coefficient(gt, pred)
        m_ap = compute_average_precision(gt, pred)
        hd = compute_hausdorff_distance(gt, pred)

    return dice, m_ap, hd

# 케이스 ID와 슬라이스 ID를 문자열에서 추출
def parse_case_and_slice_id(full_name: str):
    """
    문자열에서 케이스 ID와 슬라이스 ID를 분리.
    예: 'case0001_slice012' -> ('case0001', 12)
        'case0002'          -> ('case0002', 0)
    """
    if '_slice' in full_name:
        case_id, slice_str = full_name.split('_slice', 1)
        slice_id = int(slice_str)
    else:
        case_id = full_name
        slice_id = 0
        
    return case_id, slice_id

# 슬라이스 단위 추론 함수
def run_inference_on_slice(image: torch.Tensor, label: torch.Tensor, model: torch.nn.Module):
    """
    3개 슬라이스를 입력으로 받아 2D 예측을 수행.

    Args:
        image:  (B=1, 3, H, W) 형태의 텐서.
        label:  (B=1, H, W) 형태의 텐서.
        model:  추론할 모델(torch.nn.Module).

    Returns:
        pred_slice (np.ndarray): (H, W), 슬라이스 예측 결과.
        label_slice (np.ndarray): (H, W), 슬라이스 라벨(정답).
    """
    image_np = image.squeeze(0).cpu().numpy()  # (3, H, W)
    label_np = label.squeeze(0).cpu().numpy()  # (H, W)

    C, H, W = image_np.shape
    assert C == 3, "Input image must have 3 channels (3 slices)."

    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(image_np).unsqueeze(0).float().cuda()  # (1, 3, H, W)
        logits = model(input_tensor)   # 모델 예측 수행 (1, num_classes, H, W)
        pred_2d = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()

    prediction = pred_2d.astype(np.uint8)
    label_slice = label_np.astype(np.uint8)

    return prediction, label_slice

# 슬라이스 단위 예측 결과를 dict에 저장
def accumulate_slice_prediction(pred_dict, label_dict, case_id, slice_id, pred_2d, label_2d):
    """
    슬라이스 단위 예측 결과를 dict에 누적.

    pred_dict[case_id][slice_id] = pred_2d
    label_dict[case_id][slice_id] = label_2d
    """
    if case_id not in pred_dict:
        pred_dict[case_id] = {}
        label_dict[case_id] = {}
        
    pred_dict[case_id][slice_id] = pred_2d
    label_dict[case_id][slice_id] = label_2d

# 슬라이스들을 합쳐 3D 볼륨 생성
def build_3d_volume(pred_slices: dict, label_slices: dict):
    """
    슬라이스 딕셔너리에서 3D 볼륨 생성.
    {slice_id: 2D pred, ...} -> 3D 배열로 변환.

    Returns:
        pred_3d: (H, W, depth), 예측 볼륨.
        label_3d: (H, W, depth), 라벨 볼륨.
    """
    sorted_ids = sorted(pred_slices.keys())
    min_z, max_z = sorted_ids[0], sorted_ids[-1]
    depth = max_z - min_z + 1

    # H, W 파악
    any_slice = sorted_ids[0]
    H, W = pred_slices[any_slice].shape

    pred_3d = np.zeros((H, W, depth), dtype=np.uint8)
    label_3d = np.zeros((H, W, depth), dtype=np.uint8)

    for z in sorted_ids:
        pred_3d[..., z - min_z] = pred_slices[z]
        label_3d[..., z - min_z] = label_slices[z]

    return pred_3d, label_3d

# 전체 추론 및 메트릭 계산
def inference(args, model, test_save_path=None):
    # 1) DataLoader 준비
    test_transform = T.Compose([
        Resize(output_size=[args.img_size, args.img_size]),
        ToTensor()  # (H, W, 3) -> (3, H, W)
    ])
    
    db_test = COCA_dataset(
        base_dir=args.volume_path,
        split="test",
        list_dir=args.list_dir,
        transform=test_transform
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    # 2) 슬라이스 예측 결과를 저장할 dict 생성
    pred_slices_dict = {}
    label_slices_dict = {}

    # 3) 슬라이스 단위 추론 및 dict 누적
    for i_batch, sampled_batch in tqdm(enumerate(testloader, start=1), total=len(testloader)):
        # 이미지와 라벨 로드
        image, label, full_case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        # 케이스 ID 및 슬라이스 ID 추출
        case_id, slice_id = parse_case_and_slice_id(full_case_name)

        # 슬라이스 추론
        pred_2d, label_2d = run_inference_on_slice(image, label, model)

        # 결과 누적
        accumulate_slice_prediction(pred_slices_dict, label_slices_dict, case_id, slice_id, pred_2d, label_2d)

    # 4) 케이스 단위로 3D 볼륨 합치기 및 메트릭 계산 (3D 평가)
    metric_list_per_case = []
    case_list = sorted(pred_slices_dict.keys())

    for case_id in case_list:
        # 3D 볼륨 생성
        pred_3d, label_3d = build_3d_volume(pred_slices_dict[case_id], label_slices_dict[case_id])
        
        pred_3d = np.transpose(pred_3d, (2, 0, 1))   # (depth, H, W)
        label_3d = np.transpose(label_3d, (2, 0, 1)) # (depth, H, W)
        
        if args.is_savenii and test_save_path is not None:
            # NIfTI 파일 저장
            pred_itk  = sitk.GetImageFromArray(pred_3d.astype(np.float32))
            label_itk = sitk.GetImageFromArray(label_3d.astype(np.float32))

            pred_itk.SetSpacing((0.375, 0.375, args.z_spacing))
            label_itk.SetSpacing((0.375, 0.375, args.z_spacing))

            sitk.WriteImage(pred_itk,  f"{test_save_path}/{case_id}_pred.nii.gz")
            sitk.WriteImage(label_itk, f"{test_save_path}/{case_id}_gt.nii.gz")

        # 클래스별 Dice, mAP, HD 계산
        metrics_this_case = []
        for c in range(1, args.num_classes):
            dice, m_ap, hd = calculate_metric_percase(pred_3d == c, label_3d == c)
            metrics_this_case.append((dice, m_ap, hd))
        
        metrics_this_case = np.array(metrics_this_case)
        metric_list_per_case.append(metrics_this_case)

        # 케이스별 평균 메트릭 로깅
        mean_dice_case = np.nanmean(metrics_this_case[:, 0])
        mean_map_case  = np.nanmean(metrics_this_case[:, 1])
        mean_hd_case   = np.nanmean(metrics_this_case[:, 2])
        logging.info(f"{case_id} - Dice: {mean_dice_case:.4f}, mAP: {mean_map_case:.4f}, HD95: {mean_hd_case:.2f}")

    # 5) 전체 평균 메트릭 계산 (3D 평가)
    metric_array = np.array(metric_list_per_case)

    # 클래스별 평균 계산
    logging.info(f"\n")
    for c_idx in range(1, args.num_classes):
        dice_c = np.nanmean(metric_array[:, c_idx-1, 0])
        map_c  = np.nanmean(metric_array[:, c_idx-1, 1])
        hd_c   = np.nanmean(metric_array[:, c_idx-1, 2])
        logging.info(f"[3D] Class {c_idx} - Dice: {dice_c:.4f}, mAP: {map_c:.4f}, HD95: {hd_c:.2f}")

    # 전체 평균
    mean_dice_all = np.nanmean(metric_array[:, :, 0])
    mean_map_all  = np.nanmean(metric_array[:, :, 1])
    mean_hd_all   = np.nanmean(metric_array[:, :, 2])
    logging.info(f"[3D] Testing performance - Mean Dice: {mean_dice_all:.4f}, Mean mAP: {mean_map_all:.4f}, Mean HD95: {mean_hd_all:.2f}")
    logging.info(f"\n")

    ### 2D LesionOnly 평가 (특정 Class의 병변이 있는 슬라이스만 평가)###
    lesion_slice_metrics = {c: [] for c in range(1, args.num_classes)}

    for case_id in case_list:
        # 케이스의 모든 슬라이스 순회
        sorted_slice_ids = sorted(pred_slices_dict[case_id].keys())
        for slice_id in sorted_slice_ids:
            pred_2d = pred_slices_dict[case_id][slice_id]
            label_2d = label_slices_dict[case_id][slice_id]
            
            for c in range(1, args.num_classes):
                # 병변 존재 여부 확인
                gt_mask = (label_2d == c)
                if gt_mask.sum() > 0:
                    pr_mask = (pred_2d == c)
                    dice_2d, map_2d, _ = calculate_metric_percase(pr_mask, gt_mask)
                    lesion_slice_metrics[c].append((dice_2d, map_2d))

    # 클래스별 병변 존재 슬라이스 평균 계산
    all_classes_dice = []
    all_classes_map = []
    for c in range(1, args.num_classes):
        metrics_c = np.array(lesion_slice_metrics[c])
        if len(metrics_c) == 0:
            logging.info(f"[2D, LesionOnly] Class {c}: no lesion slice found.")
            continue

        dice_mean = np.nanmean(metrics_c[:, 0])
        map_mean  = np.nanmean(metrics_c[:, 1])
        all_classes_dice.append(dice_mean)
        all_classes_map.append(map_mean)
        logging.info(f"[2D, LesionOnly] (#slices: {len(metrics_c)}) Class {c} - Dice: {dice_mean:.4f}, mAP: {map_mean:.4f}")

    if len(all_classes_dice) > 0:
        mean_dice_2d = np.mean(all_classes_dice)
        mean_map_2d  = np.mean(all_classes_map)
        logging.info(f"[2D, LesionOnly] Testing performance - Mean Dice: {mean_dice_2d:.4f}, Mean mAP: {mean_map_2d:.4f}")
        logging.info(f"\n")
    else:
        logging.info("[2D, LesionOnly] No lesion slices found for any class.")

    ### 2D LesionAny 평가 (Background 제외, 병변이 있는 슬라이스만 평가) ###
    lesion_any_slice_metrics = {c: [] for c in range(1, args.num_classes)}
    total_lesion_slices = 0

    for case_id in case_list:
        # 케이스의 모든 슬라이스 순회
        sorted_slice_ids = sorted(pred_slices_dict[case_id].keys())
        for slice_id in sorted_slice_ids:
            pred_2d = pred_slices_dict[case_id][slice_id]
            label_2d = label_slices_dict[case_id][slice_id]
            
            # 어떤 클래스라도 병변이 있는지 확인
            if np.any(label_2d > 0):
                total_lesion_slices += 1
                for c in range(1, args.num_classes):
                    gt_mask = (label_2d == c)
                    pr_mask = (pred_2d == c)
                    dice_2d, map_2d, _ = calculate_metric_percase(pr_mask, gt_mask)
                    lesion_any_slice_metrics[c].append((dice_2d, map_2d))

    # 클래스별 병변이 있는 모든 슬라이스 평균 계산
    any_classes_dice = []
    any_classes_map = []
    for c in range(1, args.num_classes):
        metrics_c = np.array(lesion_any_slice_metrics[c])
        if len(metrics_c) == 0:
            logging.info(f"[2D, LesionAny] Class {c}: no lesion slice found.")
            continue

        dice_mean = np.nanmean(metrics_c[:, 0])
        map_mean  = np.nanmean(metrics_c[:, 1])
        any_classes_dice.append(dice_mean)
        any_classes_map.append(map_mean)
        logging.info(f"[2D, LesionAny] (#slices: {len(metrics_c)}) Class {c} - Dice: {dice_mean:.4f}, mAP: {map_mean:.4f}")

    if len(any_classes_dice) > 0:
        mean_dice_any_2d = np.mean(any_classes_dice)
        mean_map_any_2d  = np.mean(any_classes_map)
        logging.info(f"[2D, LesionAny] Testing performance - Mean Dice: {mean_dice_any_2d:.4f}, Mean mAP: {mean_map_any_2d:.4f}")
    else:
        logging.info("[2D, LesionAny] No lesion slices found for any class.")

    return "Testing Finished!"