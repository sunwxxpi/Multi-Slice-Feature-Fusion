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

def compute_dice_coefficient(mask_gt, mask_pred):
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum

def compute_average_precision(mask_gt, mask_pred):
    precision, recall, _ = precision_recall_curve(mask_gt.flatten(), mask_pred.flatten())
    return auc(recall, precision)

def compute_hausdorff_distance(mask_gt, mask_pred):
    gt_points = np.transpose(np.nonzero(mask_gt))
    pred_points = np.transpose(np.nonzero(mask_pred))
    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.NaN
    hd_1 = directed_hausdorff(gt_points, pred_points)[0]
    hd_2 = directed_hausdorff(pred_points, gt_points)[0]
    return max(hd_1, hd_2)

def calculate_metric_percase(pred, gt):
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

def parse_case_and_slice_id(full_name: str):
    """
    'case0001_slice012' -> ('case0001', 12)
    'case0002'          -> ('case0002', 0)
    """
    if '_slice' in full_name:
        case_id, slice_str = full_name.split('_slice', 1)
        slice_id = int(slice_str)
    else:
        case_id = full_name
        slice_id = 0
    return case_id, slice_id

def run_inference_on_slice(image: torch.Tensor, label: torch.Tensor, model: torch.nn.Module,):
    """
    슬라이스 단위로 추론하는 함수.
    (3장 슬라이스를 채널로 쌓은 2D 입력 -> 모델 예측 -> 2D 결과 반환)

    Args:
        image:  (B=1, 3, H, W) 형태의 텐서.
        label:  (B=1, H, W) 형태의 텐서.
        model:  추론할 모델(torch.nn.Module).

    Returns:
        pred_slice (np.ndarray): (H, W), 슬라이스 예측 결과(0~N).
        label_slice (np.ndarray): (H, W), 슬라이스 라벨(정답).
    """
    image_np = image.squeeze(0).cpu().numpy()  # (3, H, W)
    label_np = label.squeeze(0).cpu().numpy()  # (H, W)

    C, H, W = image_np.shape
    assert C == 3, "Input image must have 3 channels (3-slices)."

    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(image_np).unsqueeze(0).float().cuda()  # (1, 3, H', W')
        logits = model(input_tensor)   # (1, num_classes, H', W') 가정
        pred_2d = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()

    prediction = pred_2d.astype(np.uint8)
    label_slice = label_np.astype(np.uint8)

    return prediction, label_slice

def accumulate_slice_prediction(pred_dict, label_dict, case_id, slice_id, pred_2d, label_2d):
    """
    dict에 슬라이스 단위 예측 결과/라벨을 누적.
    pred_dict[case_id][slice_id] = pred_2d
    label_dict[case_id][slice_id] = label_2d
    """
    if case_id not in pred_dict:
        pred_dict[case_id] = {}
        label_dict[case_id] = {}
    pred_dict[case_id][slice_id] = pred_2d
    label_dict[case_id][slice_id] = label_2d
    
def build_3d_volume(pred_slices: dict, label_slices: dict):
    """
    {slice_id: 2D pred, ...} -> 3D (H, W, depth) 배열로 변환
    """
    sorted_ids = sorted(pred_slices.keys())  # [0,1,2,...]
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

def inference(args, model, test_save_path=None):
    # 1) DataLoader 준비
    test_transform = T.Compose([
        Resize(output_size=[args.img_size, args.img_size]),
        ToTensor()  # (H,W,3) -> (3,H,W)
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

    # 2) (케이스 -> {슬라이스 -> (pred2D, label2D)}) 저장할 dict
    pred_slices_dict = {}
    label_slices_dict = {}

    # 3) 슬라이스 단위 추론 & dict 누적
    for i_batch, sampled_batch in tqdm(enumerate(testloader, start=1), total=len(testloader)):
        # image: (1,3,H,W), label: (1,H,W)
        image, label, full_case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        # case_id, slice_id 추출
        case_id, slice_id = parse_case_and_slice_id(full_case_name)

        # 슬라이스 추론
        pred_2d, label_2d = run_inference_on_slice(image, label, model)

        # 누적
        accumulate_slice_prediction(pred_slices_dict, label_slices_dict, case_id, slice_id, pred_2d, label_2d)

    # 4) 케이스 단위로 3D 볼륨 합치고, 메트릭 계산
    metric_list_per_case = []
    case_list = sorted(pred_slices_dict.keys())

    for case_id in case_list:
        # (H, W, depth)
        pred_3d, label_3d = build_3d_volume(pred_slices_dict[case_id], label_slices_dict[case_id])
        
        pred_3d = np.transpose(pred_3d, (2, 0, 1))   # (depth, H, W)
        label_3d = np.transpose(label_3d, (2, 0, 1)) # (depth, H, W)
        
        if args.is_savenii and test_save_path is not None:
            pred_itk  = sitk.GetImageFromArray(pred_3d.astype(np.float32))
            label_itk = sitk.GetImageFromArray(label_3d.astype(np.float32))

            pred_itk.SetSpacing((0.375, 0.375, args.z_spacing))
            label_itk.SetSpacing((0.375, 0.375, args.z_spacing))

            sitk.WriteImage(pred_itk,  f"{test_save_path}/{case_id}_pred.nii.gz")
            sitk.WriteImage(label_itk, f"{test_save_path}/{case_id}_gt.nii.gz")

        # 클래스별 Dice, mAP, HD
        metrics_this_case = []
        for c in range(1, args.num_classes):
            dice, m_ap, hd = calculate_metric_percase(pred_3d == c, label_3d == c)
            metrics_this_case.append((dice, m_ap, hd))
        
        metrics_this_case = np.array(metrics_this_case)  # (num_classes-1, 3)
        metric_list_per_case.append(metrics_this_case)

        # 케이스별 평균 로깅
        mean_dice_case = np.nanmean(metrics_this_case[:, 0])
        mean_map_case  = np.nanmean(metrics_this_case[:, 1])
        mean_hd_case   = np.nanmean(metrics_this_case[:, 2])
        logging.info(
            f"{case_id} - mean_dice: {mean_dice_case:.4f}, mean_m_ap: {mean_map_case:.4f}, mean_hd95: {mean_hd_case:.2f}"
        )

    # 5) 전체 케이스 평균
    metric_array = np.array(metric_list_per_case)  # (n_cases, num_classes-1, 3)

    # 클래스별 평균
    for c_idx in range(1, args.num_classes):
        dice_c = np.nanmean(metric_array[:, c_idx-1, 0])
        map_c  = np.nanmean(metric_array[:, c_idx-1, 1])
        hd_c   = np.nanmean(metric_array[:, c_idx-1, 2])
        logging.info(
            f"Mean class {c_idx} - dice: {dice_c:.4f}, m_ap: {map_c:.4f}, hd95: {hd_c:.2f}"
        )

    # 전체 평균
    mean_dice_all = np.nanmean(metric_array[:, :, 0])
    mean_map_all  = np.nanmean(metric_array[:, :, 1])
    mean_hd_all   = np.nanmean(metric_array[:, :, 2])
    logging.info(
        f"Testing performance - mean_dice: {mean_dice_all:.4f}, mean_m_ap: {mean_map_all:.4f}, mean_hd95: {mean_hd_all:.2f}"
    )

    return "Testing Finished!"