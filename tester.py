import logging
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.metrics import precision_recall_curve, auc
from scipy.ndimage import zoom
from scipy.spatial.distance import directed_hausdorff
from datasets.dataset import COCA_dataset, ToTensor

def compute_dice_coefficient(mask_gt, mask_pred):
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    return 2 * (mask_gt & mask_pred).sum() / volume_sum

def compute_average_precision(mask_gt, mask_pred):
    precision, recall, _ = precision_recall_curve(mask_gt.flatten(), mask_pred.flatten())
    return auc(recall, precision)

def compute_hausdorff_distance(mask_gt, mask_pred):
    gt_points = np.transpose(np.nonzero(mask_gt))
    pred_points = np.transpose(np.nonzero(mask_pred))
    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.NaN
    return max(directed_hausdorff(gt_points, pred_points)[0],
               directed_hausdorff(pred_points, gt_points)[0])

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if gt.sum() == 0 and pred.sum() == 0:
        return 1, 1, 0
    if gt.sum() == 0:
        return 0, 0, np.NaN
    
    return (
        compute_dice_coefficient(gt, pred),
        compute_average_precision(gt, pred),
        compute_hausdorff_distance(gt, pred)
    )

def process_slice(slice_2d, model, patch_size):
    x, y = slice_2d.shape
    if (x, y) != tuple(patch_size):
        slice_2d = zoom(slice_2d, (patch_size[0] / x, patch_size[1] / y), order=3)

    input_tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).float().cuda()
    with torch.no_grad():
        outputs = model(input_tensor)
        out_2d = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()

    if (x, y) != tuple(patch_size):
        out_2d = zoom(out_2d, (x / patch_size[0], y / patch_size[1]), order=0)

    return out_2d

def compute_metrics_2d(out_2d, label_2d, classes):
    lesiononly_metrics = {c: [] for c in range(1, classes)}
    lesionany_metrics = {c: [] for c in range(1, classes)}

    for c in range(1, classes):
        gt_mask = (label_2d == c)
        if gt_mask.sum() > 0:
            pred_mask = (out_2d == c)
            lesiononly_metrics[c].append(calculate_metric_percase(pred_mask.copy(), gt_mask.copy())[:2])

        if label_2d.sum() > 0:
            pred_mask = (out_2d == c)
            lesionany_metrics[c].append(calculate_metric_percase(pred_mask.copy(), gt_mask.copy())[:2])

    return lesiononly_metrics, lesionany_metrics

def log_2d_metrics(tag, metrics_all, num_classes):
    class_dice, class_map = [], []
    for c in range(1, num_classes):
        metrics = np.array(metrics_all[c])
        if metrics.size == 0:
            logging.info(f"{tag} Class {c}: no lesion slice found.")
            continue

        dice_mean, map_mean = np.nanmean(metrics[:, 0]), np.nanmean(metrics[:, 1])
        class_dice.append(dice_mean)
        class_map.append(map_mean)

        logging.info(f"{tag} (#slices: {len(metrics)}) Class {c} - Dice: {dice_mean:.4f}, mAP: {map_mean:.4f}")

    if class_dice:
        logging.info(f"{tag} Mean Dice: {np.mean(class_dice):.4f}, Mean mAP: {np.mean(class_map):.4f}")
        logging.info(f"\n")
    else:
        logging.info(f"{tag} No lesion slices found for any class.")

def test_single_volume(image, label, model, classes, patch_size, test_save_path=None, case=None, z_spacing=1):
    image_np, label_np = image.squeeze().cpu().detach().numpy(), label.squeeze().cpu().detach().numpy()
    D, H, W = image_np.shape
    prediction_3d = np.zeros_like(label_np, dtype=np.uint8)

    lesiononly_slice_metrics, lesionany_slice_metrics = {c: [] for c in range(1, classes)}, {c: [] for c in range(1, classes)}

    model.eval()
    for d in range(D):
        slice_2d, label_2d = image_np[d], label_np[d]
        out_2d = process_slice(slice_2d, model, patch_size)
        prediction_3d[d] = out_2d

        lesiononly, lesionany = compute_metrics_2d(out_2d, label_2d, classes)
        for c in range(1, classes):
            lesiononly_slice_metrics[c].extend(lesiononly[c])
            lesionany_slice_metrics[c].extend(lesionany[c])

    metric_list_3d = [
        calculate_metric_percase((prediction_3d == c).astype(np.uint8), (label_np == c).astype(np.uint8))
        for c in range(1, classes)
    ]

    if test_save_path and case:
        for array, suffix in zip([image_np, prediction_3d, label_np], ["img", "pred", "gt"]):
            itk_img = sitk.GetImageFromArray(array.astype(np.float32))
            itk_img.SetSpacing((0.375, 0.375, z_spacing))
            sitk.WriteImage(itk_img, f"{test_save_path}/{case}_{suffix}.nii.gz")

    return metric_list_3d, lesiononly_slice_metrics, lesionany_slice_metrics

def inference(args, model, test_save_path=None):
    test_transform = T.Compose([ToTensor()])
    testloader = DataLoader(
        COCA_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test", transform=test_transform),
        batch_size=1, shuffle=False, num_workers=1
    )

    logging.info(f"{len(testloader)} test iterations per epoch")
    model.eval()

    metrics_3d_all = []
    lesiononly_metrics_all, lesionany_metrics_all = {c: [] for c in range(1, args.num_classes)}, {c: [] for c in range(1, args.num_classes)}

    for sampled_batch in testloader:
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metrics_3d, lesiononly_metrics, lesionany_metrics = test_single_volume(
            image, label, model, args.num_classes, [args.img_size, args.img_size], test_save_path, case_name, args.z_spacing
        )

        metrics_3d_all.append(metrics_3d)
        for c in range(1, args.num_classes):
            lesiononly_metrics_all[c].extend(lesiononly_metrics[c])
            lesionany_metrics_all[c].extend(lesionany_metrics[c])

        logging.info(f"{case_name} - Dice: {np.nanmean(metrics_3d, axis=0)[0]:.4f}, mAP: {np.nanmean(metrics_3d, axis=0)[1]:.4f}, HD95: {np.nanmean(metrics_3d, axis=0)[2]:.2f}")

    metrics_3d_array = np.array(metrics_3d_all)

    logging.info(f"\n")
    for c in range(1, args.num_classes):
        class_metrics = metrics_3d_array[:, c - 1]
        logging.info(f"[3D] Class {c} - Dice: {np.nanmean(class_metrics[:, 0]):.4f}, mAP: {np.nanmean(class_metrics[:, 1]):.4f}, HD95: {np.nanmean(class_metrics[:, 2]):.2f}")

    logging.info(f"[3D] Mean Dice: {np.nanmean(metrics_3d_array[:, :, 0]):.4f}, Mean mAP: {np.nanmean(metrics_3d_array[:, :, 1]):.4f}, Mean HD95: {np.nanmean(metrics_3d_array[:, :, 2]):.2f}")
    logging.info(f"\n")

    log_2d_metrics("[2D, LesionOnly]", lesiononly_metrics_all, args.num_classes)
    log_2d_metrics("[2D, LesionAny]", lesionany_metrics_all, args.num_classes)

    return "Testing Finished!"