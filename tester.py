import logging
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from scipy.ndimage import zoom
from scipy.spatial.distance import directed_hausdorff
from datasets.dataset import COCA_dataset, ToTensor

def compute_dice_coefficient(mask_gt, mask_pred):
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum

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
        return 1, 0
    elif gt.sum() == 0 and pred.sum() > 0:
        return 0, np.NaN
    else:
        return (compute_dice_coefficient(gt, pred), compute_hausdorff_distance(gt, pred))

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

def test_single_volume(image, label, model, classes, patch_size, test_save_path=None, case=None, z_spacing=1):
    image_np, label_np = image.squeeze().cpu().detach().numpy(), label.squeeze().cpu().detach().numpy()
    D, H, W = image_np.shape
    prediction_3d = np.zeros_like(label_np, dtype=np.uint8)

    model.eval()
    for d in range(D):
        slice_2d = image_np[d]
        out_2d = process_slice(slice_2d, model, patch_size)
        prediction_3d[d] = out_2d

    metric_list_3d = [
        calculate_metric_percase((prediction_3d == c).astype(np.uint8), (label_np == c).astype(np.uint8))
        for c in range(1, classes)
    ]

    if test_save_path and case:
        for array, suffix in zip([image_np, prediction_3d, label_np], ["img", "pred", "gt"]):
            itk_img = sitk.GetImageFromArray(array.astype(np.float32))
            itk_img.SetSpacing((0.375, 0.375, z_spacing))
            sitk.WriteImage(itk_img, f"{test_save_path}/{case}_{suffix}.nii.gz")

    return metric_list_3d

def inference(args, model, test_save_path=None):
    test_transform = T.Compose([ToTensor()])
    db_test = COCA_dataset(base_dir=args.volume_path,
                           list_dir=args.list_dir,
                           split="test",
                           transform=test_transform)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"{len(testloader)} test iterations per epoch")

    metrics_3d_all = []

    for sampled_batch in testloader:
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metrics_3d = test_single_volume(image, label, model, 
                                        args.num_classes, [args.img_size, args.img_size], 
                                        test_save_path, case_name, args.z_spacing)

        metrics_3d_all.append(metrics_3d)

        logging.info(f"{case_name} - Dice: {np.nanmean(metrics_3d, axis=0)[0]:.4f}, HD95: {np.nanmean(metrics_3d, axis=0)[1]:.2f}")

    metrics_3d_array = np.array(metrics_3d_all)

    logging.info(f"\n")
    for c in range(1, args.num_classes):
        class_metrics = metrics_3d_array[:, c - 1]
        logging.info(f"[3D] Class {c} - Dice: {np.nanmean(class_metrics[:, 0]):.4f}, HD95: {np.nanmean(class_metrics[:, 1]):.2f}")

    logging.info(f"[3D] Mean Dice: {np.nanmean(metrics_3d_array[:, :, 0]):.4f}, Mean HD95: {np.nanmean(metrics_3d_array[:, :, 1]):.2f}")
    logging.info(f"\n")

    return "Testing Finished!"