import logging
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms as T
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from datasets.dataset import COCA_dataset, Resize, ToTensor

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
        with autocast():
            input_tensor = image.float().cuda()
            logits = model(input_tensor)
            pred_2d = torch.argmax(torch.softmax(logits[3], dim=1), dim=1).squeeze(0).cpu().numpy()

    prediction = pred_2d.astype(np.uint8)
    label_slice = label_np.astype(np.uint8)
    
    return prediction, label_slice

def accumulate_slice_prediction(image_dict: dict, pred_dict: dict, label_dict: dict, case_id: str, slice_id: int, pred_2d: np.ndarray, label_2d: np.ndarray, img_2d: np.ndarray):
    if case_id not in pred_dict:
        image_dict[case_id] = {}
        pred_dict[case_id] = {}
        label_dict[case_id] = {}
        
    image_dict[case_id][slice_id] = img_2d
    pred_dict[case_id][slice_id] = pred_2d
    label_dict[case_id][slice_id] = label_2d

def build_3d_volume(image_slices: dict, pred_slices: dict, label_slices: dict, case_id, args, test_save_path) -> tuple:
    sorted_ids = sorted(pred_slices.keys())
    min_z, max_z = sorted_ids[0], sorted_ids[-1]
    depth = max_z - min_z + 1

    any_slice = sorted_ids[0]
    H, W = pred_slices[any_slice].shape

    image_3d = np.zeros((depth, H, W), dtype=np.uint8)
    pred_3d = np.zeros((depth, H, W), dtype=np.uint8)
    label_3d = np.zeros((depth, H, W), dtype=np.uint8)

    for z in sorted_ids:
        index = z - min_z
        image_3d[index, :, :] = image_slices[z][1, :, :]
        pred_3d[index, :, :] = pred_slices[z]
        label_3d[index, :, :] = label_slices[z]
        
    if args.is_savenii and test_save_path:
        for array, suffix in zip([image_3d, pred_3d, label_3d], ["img", "pred", "gt"]):
            array = np.flip(np.transpose(array, (0, 2, 1)), (1, 2))
            arr_itk = sitk.GetImageFromArray(array.astype(np.float32))
            arr_itk.SetSpacing((0.375, 0.375, args.z_spacing))
            sitk.WriteImage(arr_itk, f"{test_save_path}/{case_id}_{suffix}.nii.gz")

    return image_3d, pred_3d, label_3d

def compute_metrics_3d(pred_3d, label_3d, num_classes, dice_metric, miou_metric, hd_metric, case_id):
    pred_tensor = torch.from_numpy(pred_3d).unsqueeze(0).unsqueeze(0).float().cuda()
    label_tensor = torch.from_numpy(label_3d).unsqueeze(0).unsqueeze(0).float().cuda()
    
    metrics_per_class = []
    
    logging.info(f"Metrics for Case: {case_id}")
    for c in range(1, num_classes):
        pred_class = (pred_tensor == c).float()
        label_class = (label_tensor == c).float()
        
        dice = dice_metric(pred_class, label_class)
        miou = miou_metric(pred_class, label_class)
        hd = hd_metric(pred_class, label_class)
        
        if isinstance(hd, torch.Tensor):
            hd = hd.item()

        gt_sum = label_class.sum()
        pred_sum = pred_class.sum()

        if gt_sum == 0 and pred_sum == 0:
            status = "(GT==0 & Pred==0)"
        elif gt_sum == 0 and pred_sum > 0:
            status = "(GT==0 & Pred>0)"
        elif gt_sum > 0 and pred_sum == 0:
            hd = np.nan
            status = "(GT>0 & Pred==0)"
        else:
            status = "(GT>0 & Pred>0)"

        logging.info(f"  Class {c}: Dice: {dice.item():.4f}, mIoU: {miou.item():.4f}, HD: {hd:.2f} {status}")
        
        metrics_per_class.append((dice.item(), miou.item(), hd))
    
    metrics_per_class = np.array(metrics_per_class)
    mean_dice = np.nanmean(metrics_per_class[:, 0])
    mean_iou = np.nanmean(metrics_per_class[:, 1])
    mean_hd = np.nanmean(metrics_per_class[:, 2])
    
    logging.info(f"  [Case {case_id}] - Mean Dice: {mean_dice:.4f}, Mean mIoU: {mean_iou:.4f}, Mean HD: {mean_hd:.2f}\n")
    
    return metrics_per_class

def log_3d_metrics(metric_array, num_classes):
    logging.info("\nOverall 3D Metrics Across All Cases:")

    class_dice_means = []
    class_miou_means = []
    class_hd_means = []

    for c_idx in range(1, num_classes):
        dice_c = np.nanmean(metric_array[:, c_idx-1, 0])
        miou_c = np.nanmean(metric_array[:, c_idx-1, 1])
        hd_c = np.nanmean(metric_array[:, c_idx-1, 2])

        logging.info(f"  [3D] Class {c_idx} - Dice: {dice_c:.4f}, mIoU: {miou_c:.4f}, HD: {hd_c:.2f}")

        class_dice_means.append(dice_c)
        class_miou_means.append(miou_c)
        class_hd_means.append(hd_c)

    mean_dice_all = np.nanmean(class_dice_means)
    mean_miou_all = np.nanmean(class_miou_means)
    mean_hd_all = np.nanmean(class_hd_means)

    logging.info(f"  [3D] Testing Performance - Mean Dice: {mean_dice_all:.4f}, Mean mIoU: {mean_miou_all:.4f}, Mean HD: {mean_hd_all:.2f}")

def inference(args, model, test_save_path: str = None):
    test_transform = T.Compose([Resize(output_size=[args.img_size, args.img_size]),
                                ToTensor()])
    db_test = COCA_dataset(base_dir=args.root_path,
                           list_dir=args.list_dir,
                           split="test",
                           transform=test_transform)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"{len(testloader)} test iterations per epoch")

    image_slices_dict = {}
    pred_slices_dict = {}
    label_slices_dict = {}

    for sampled_batch in tqdm(testloader, total=len(testloader), desc="Inference"):
        image, label, full_case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        case_id, slice_id = parse_case_and_slice_id(full_case_name)
        img_2d = image.squeeze(0).cpu().numpy()
        pred_2d, label_2d = run_inference_on_slice(image, label, model)
        accumulate_slice_prediction(image_slices_dict, pred_slices_dict, label_slices_dict, case_id, slice_id, pred_2d, label_2d, img_2d)

    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=True)
    miou_metric = MeanIoU(include_background=True, reduction="mean", ignore_empty=True)
    hd_metric = HausdorffDistanceMetric(include_background=True, distance_metric="euclidean", percentile=95)

    metric_list_per_case = []
    
    case_list = sorted(pred_slices_dict.keys())
    for case_id in case_list:
        _, pred_3d, label_3d = build_3d_volume(image_slices_dict[case_id], pred_slices_dict[case_id], label_slices_dict[case_id], case_id, args, test_save_path)
        metrics_per_case = compute_metrics_3d(pred_3d, label_3d, args.num_classes, dice_metric, miou_metric, hd_metric, case_id)
        metric_list_per_case.append(metrics_per_case)

    metric_array = np.array(metric_list_per_case)
    log_3d_metrics(metric_array, args.num_classes)

    return "Testing Finished!"