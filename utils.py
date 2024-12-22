import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import precision_recall_curve, auc
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import zoom


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        
        # Background(index 0) 제외, Foreground(나머지 클래스)에 대해서만 Dice Loss 계산
        for i in range(1, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        
        # Background를 제외한 클래스 수로 나눠서 평균 계산
        return loss / (self.n_classes - 1)


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


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
                
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((0.375, 0.375, z_spacing))
        prd_itk.SetSpacing((0.375, 0.375, z_spacing))
        lab_itk.SetSpacing((0.375, 0.375, z_spacing))
        sitk.WriteImage(prd_itk, f"{test_save_path}/{case}_pred.nii.gz")
        sitk.WriteImage(img_itk, f"{test_save_path}/{case}_img.nii.gz")
        sitk.WriteImage(lab_itk, f"{test_save_path}/{case}_gt.nii.gz")

    return metric_list