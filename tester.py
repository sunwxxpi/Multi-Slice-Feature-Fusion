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

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image_np = image.squeeze().cpu().detach().numpy()
    label_np = label.squeeze().cpu().detach().numpy()

    D, H, W = image_np.shape
    prediction_3d = np.zeros_like(label_np)

    for d in range(D):
        slice_2d = image_np[d, ...]

        x, y = slice_2d.shape
        if x != patch_size[0] or y != patch_size[1]:
            slice_2d = zoom(slice_2d, (patch_size[0]/x, patch_size[1]/y), order=3)

        input_tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).float().cuda()

        net.eval()
        with torch.no_grad():
            outputs = net(input_tensor)  
            out_2d = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()
        
        if x != patch_size[0] or y != patch_size[1]:
            out_2d = zoom(out_2d, (x/patch_size[0], y/patch_size[1]), order=0)

        prediction_3d[d, ...] = out_2d

    metric_list = []
    for c in range(1, classes):
        dice, m_ap, hd = calculate_metric_percase(prediction_3d == c, label_np == c)
        metric_list.append((dice, m_ap, hd))

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

    return metric_list

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
    
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    
    metric_list_all = []
    
    for i_batch, sampled_batch in tqdm(enumerate(testloader, start=1)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list_all.append(metric_i)
        
        mean_dice_case = np.nanmean(metric_i, axis=0)[0]
        mean_m_ap_case = np.nanmean(metric_i, axis=0)[1]
        mean_hd95_case = np.nanmean(metric_i, axis=0)[2]
        logging.info('%s - mean_dice: %.4f, mean_m_ap: %.4f, mean_hd95: %.2f' % (case_name, mean_dice_case, mean_m_ap_case, mean_hd95_case))
    
    metric_array = np.array(metric_list_all)
    
    for i in range(1, args.num_classes):
        class_dice = np.nanmean(metric_array[:, i-1, 0])
        class_m_ap = np.nanmean(metric_array[:, i-1, 1])
        class_hd95 = np.nanmean(metric_array[:, i-1, 2])
        logging.info('Mean class %d - mean_dice: %.4f, mean_m_ap: %.4f, mean_hd95: %.2f' % (i, class_dice, class_m_ap, class_hd95))
        
    mean_dice = np.nanmean(metric_array[:,:,0])
    mean_m_ap = np.nanmean(metric_array[:,:,1])
    mean_hd95 = np.nanmean(metric_array[:,:,2])
    
    logging.info('Testing performance in best val model - mean_dice : %.4f, mean_m_ap : %.4f, mean_hd95 : %.2f' % (mean_dice, mean_m_ap, mean_hd95))
    
    return "Testing Finished!"