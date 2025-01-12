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

def compute_dice_coefficient(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum

def compute_average_precision(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(mask_gt.flatten(), mask_pred.flatten())
    return auc(recall, precision)

def compute_hausdorff_distance(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    gt_points = np.transpose(np.nonzero(mask_gt))
    pred_points = np.transpose(np.nonzero(mask_pred))
    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.NaN
    hd_forward = directed_hausdorff(gt_points, pred_points)[0]
    hd_backward = directed_hausdorff(pred_points, gt_points)[0]
    return max(hd_forward, hd_backward)

def calculate_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple:
    pred_binary = (pred > 0).astype(np.uint8)
    gt_binary = (gt > 0).astype(np.uint8)

    if gt_binary.sum() == 0 and pred_binary.sum() == 0:
        return 1.0, 1.0, 0.0
    elif gt_binary.sum() == 0 and pred_binary.sum() > 0:
        return 0.0, 0.0, np.NaN
    else:
        dice = compute_dice_coefficient(gt_binary, pred_binary)
        m_ap = compute_average_precision(gt_binary, pred_binary)
        hd = compute_hausdorff_distance(gt_binary, pred_binary)
        return dice, m_ap, hd

def parse_case_and_slice_id(full_name: str) -> tuple:
    if '_slice' in full_name:
        case_id, slice_str = full_name.split('_slice', 1)
        slice_id = int(slice_str)
    else:
        case_id = full_name
        slice_id = 0
        
    return case_id, slice_id

def run_inference_on_slice(image: torch.Tensor, label: torch.Tensor, model: torch.nn.Module) -> tuple:
    image_np = image.squeeze(0).cpu().numpy()
    label_np = label.squeeze(0).cpu().numpy()
    C, H, W = image_np.shape
    assert C == 3, "Input image must have 3 channels (3 slices)."

    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(image_np).unsqueeze(0).float().cuda()
        logits = model(input_tensor)
        pred_2d = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()

    prediction = pred_2d.astype(np.uint8)
    label_slice = label_np.astype(np.uint8)
    
    return prediction, label_slice

def accumulate_slice_prediction(pred_dict: dict, label_dict: dict, case_id: str, slice_id: int, pred_2d: np.ndarray, label_2d: np.ndarray):
    if case_id not in pred_dict:
        pred_dict[case_id] = {}
        label_dict[case_id] = {}
        
    pred_dict[case_id][slice_id] = pred_2d
    label_dict[case_id][slice_id] = label_2d

def build_3d_volume(pred_slices: dict, label_slices: dict, case_id, args, test_save_path) -> tuple:
    sorted_ids = sorted(pred_slices.keys())
    min_z, max_z = sorted_ids[0], sorted_ids[-1]
    depth = max_z - min_z + 1

    any_slice = sorted_ids[0]
    H, W = pred_slices[any_slice].shape

    pred_3d = np.zeros((H, W, depth), dtype=np.uint8)
    label_3d = np.zeros((H, W, depth), dtype=np.uint8)

    for z in sorted_ids:
        index = z - min_z
        pred_3d[:, :, index] = pred_slices[z]
        label_3d[:, :, index] = label_slices[z]
        
    if args.is_savenii and test_save_path:
        pred_itk = np.transpose(pred_3d, (2, 0, 1))
        label_itk = np.transpose(label_3d, (2, 0, 1))
        
        pred_itk = sitk.GetImageFromArray(pred_itk.astype(np.float32))
        label_itk = sitk.GetImageFromArray(label_itk.astype(np.float32))
        
        spacing = (0.375, 0.375, args.z_spacing)
        pred_itk.SetSpacing(spacing)
        label_itk.SetSpacing(spacing)

        sitk.WriteImage(pred_itk, f"{test_save_path}/{case_id}_pred.nii.gz")
        sitk.WriteImage(label_itk, f"{test_save_path}/{case_id}_gt.nii.gz")

    return pred_3d, label_3d

def compute_metrics_3d(pred_3d, label_3d, num_classes, case_id):
    metrics_per_class = []
    
    for c in range(1, num_classes):
        dice, m_ap, hd = calculate_metrics(pred_3d == c, label_3d == c)
        metrics_per_class.append((dice, m_ap, hd))

    metrics_per_class = np.array(metrics_per_class)
    mean_dice = np.nanmean(metrics_per_class[:, 0])
    mean_map = np.nanmean(metrics_per_class[:, 1])
    mean_hd = np.nanmean(metrics_per_class[:, 2])

    logging.info(f"{case_id} - Dice: {mean_dice:.4f}, mAP: {mean_map:.4f}, HD95: {mean_hd:.2f}")

    return metrics_per_class

def log_3d_metrics(metric_array, num_classes):
    mean_dice_all = np.nanmean(metric_array[:, :, 0])
    mean_map_all = np.nanmean(metric_array[:, :, 1])
    mean_hd_all = np.nanmean(metric_array[:, :, 2])

    logging.info("\n")
    for c_idx in range(1, num_classes):
        dice_c = np.nanmean(metric_array[:, c_idx-1, 0])
        map_c = np.nanmean(metric_array[:, c_idx-1, 1])
        hd_c = np.nanmean(metric_array[:, c_idx-1, 2])
        logging.info(f"[3D] Class {c_idx} - Dice: {dice_c:.4f}, mAP: {map_c:.4f}, HD95: {hd_c:.2f}")

    logging.info(f"[3D] Testing performance - Mean Dice: {mean_dice_all:.4f}, Mean mAP: {mean_map_all:.4f}, Mean HD95: {mean_hd_all:.2f}")
    logging.info("\n")

def compute_metrics_2d(pred_slices, label_slices, num_classes, mode="LesionOnly"):
    slice_metrics = {c: [] for c in range(1, num_classes)}
    valid_slices = 0

    for case_id in pred_slices.keys():
        for slice_id in sorted(pred_slices[case_id].keys()):
            pred_2d = pred_slices[case_id][slice_id]
            label_2d = label_slices[case_id][slice_id]

            if mode == "LesionOnly" and not np.any(label_2d > 0):
                continue
            elif mode == "LesionAny" and not np.any(label_2d > 0):
                continue

            valid_slices += 1

            for c in range(1, num_classes):
                gt_mask = (label_2d == c)
                if mode == "LesionOnly" and gt_mask.sum() == 0:
                    continue

                pr_mask = (pred_2d == c)
                dice, m_ap, _ = calculate_metrics(pr_mask, gt_mask)
                slice_metrics[c].append((dice, m_ap))
                
    return slice_metrics

def log_2d_metrics(slice_metrics, mode):
    all_classes_dice = []
    all_classes_map = []
    
    for c, metrics in slice_metrics.items():
        metrics = np.array(metrics)
        if metrics.size == 0:
            logging.info(f"[2D, {mode}] Class {c}: no relevant slices found.")
            continue

        dice_mean = np.nanmean(metrics[:, 0])
        map_mean = np.nanmean(metrics[:, 1])
        all_classes_dice.append(dice_mean)
        all_classes_map.append(map_mean)
        logging.info(f"[2D, {mode}] (#slices: {len(metrics)}) Class {c} - Dice: {dice_mean:.4f}, mAP: {map_mean:.4f}")

    if all_classes_dice:
        mean_dice = np.mean(all_classes_dice)
        mean_map = np.mean(all_classes_map)
        logging.info(f"[2D, {mode}] Testing performance - Mean Dice: {mean_dice:.4f}, Mean mAP: {mean_map:.4f}")
        logging.info(f"\n")
    else:
        logging.info(f"[2D, {mode}] No relevant slices found for any class.")

def inference(args, model, test_save_path: str = None):
    test_transform = T.Compose([
        Resize(output_size=[args.img_size, args.img_size]),
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
    
    pred_slices_dict = {}
    label_slices_dict = {}

    for _, sampled_batch in tqdm(enumerate(testloader, start=1), total=len(testloader), desc="Inference"):
        image, label, full_case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        case_id, slice_id = parse_case_and_slice_id(full_case_name)
        pred_2d, label_2d = run_inference_on_slice(image, label, model)
        accumulate_slice_prediction(pred_slices_dict, label_slices_dict, case_id, slice_id, pred_2d, label_2d)

    metric_list_per_case = []
    
    case_list = sorted(pred_slices_dict.keys())
    for case_id in case_list:
        pred_3d, label_3d = build_3d_volume(pred_slices_dict[case_id], label_slices_dict[case_id], case_id, args, test_save_path)
        metrics_per_case = compute_metrics_3d(pred_3d, label_3d, args.num_classes, case_id)
        metric_list_per_case.append(metrics_per_case)

    metric_array = np.array(metric_list_per_case)
    log_3d_metrics(metric_array, args.num_classes)

    lesion_only_metrics = compute_metrics_2d(pred_slices_dict, label_slices_dict, args.num_classes, mode="LesionOnly")
    log_2d_metrics(lesion_only_metrics, mode="LesionOnly")

    lesion_any_metrics = compute_metrics_2d(pred_slices_dict, label_slices_dict, args.num_classes, mode="LesionAny")
    log_2d_metrics(lesion_any_metrics, mode="LesionAny")

    return "Testing Finished!"