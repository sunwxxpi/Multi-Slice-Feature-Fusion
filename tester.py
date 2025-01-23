import logging
import numpy as np
import SimpleITK as sitk
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms as T
from scipy.ndimage import zoom
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from datasets.dataset import COCA_dataset, ToTensor

def process_slice(slice_2d, model, patch_size):
    x, y = slice_2d.shape
    if (x, y) != tuple(patch_size):
        slice_2d = zoom(slice_2d, (patch_size[0] / x, patch_size[1] / y), order=3)

    input_tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).float().cuda()

    model.eval()
    with torch.no_grad():
        with autocast():
            outputs = model(input_tensor)
            out_2d = torch.argmax(torch.softmax(outputs[3], dim=1), dim=1).squeeze(0).cpu().numpy()

    if (x, y) != tuple(patch_size):
        out_2d = zoom(out_2d, (x / patch_size[0], y / patch_size[1]), order=0)

    return out_2d

def test_single_volume(image, label, model, classes, patch_size, dice_metric, miou_metric, hd_metric, test_save_path=None, case=None, z_spacing=1):
    image_np, label_np = image.squeeze().cpu().detach().numpy(), label.squeeze().cpu().detach().numpy()
    D, H, W = image_np.shape
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
            status = "(GT==0 & Pred>0)"
        elif gt_sum > 0 and pred_sum == 0:
            hd = np.nan
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
    test_transform = T.Compose([ToTensor()])
    db_test = COCA_dataset(base_dir=args.volume_path,
                           list_dir=args.list_dir,
                           split="test",
                           transform=test_transform)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"{len(testloader)} test iterations per epoch")

    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=True)
    miou_metric = MeanIoU(include_background=True, reduction="mean", ignore_empty=True)
    hd_metric = HausdorffDistanceMetric(include_background=True, distance_metric="euclidean", percentile=95)

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