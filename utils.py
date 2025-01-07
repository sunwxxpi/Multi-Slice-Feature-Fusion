import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import precision_recall_curve, auc
from scipy.spatial.distance import directed_hausdorff

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
    def __init__(self):
        super(DiceLoss, self).__init__()

    def _one_hot_encoder(self, input_tensor, n_classes):
        tensor_list = []
        for i in range(n_classes):
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
        n_classes = inputs.size(1)
        
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target, n_classes)
        if weight is None:
            weight = [1] * n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        
        # Background(index 0) 제외, Foreground(나머지 클래스)에 대해서만 Dice Loss 계산
        for i in range(1, n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        
        # Background를 제외한 클래스 수로 나눠서 평균 계산
        return loss / (n_classes - 1)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs: (N, C, H, W) 형태의 로짓(logits)
        targets: (N, H, W) 형태로, 각 픽셀이 [0, C-1] 범위를 갖는 정수 레이블
        """
        # log_softmax로 변환 -> log_pt
        log_pt = F.log_softmax(inputs, dim=1)  # (N, C, H, W)
        
        # pt = exp(log_pt)
        pt = torch.exp(log_pt)  # 확률 형태로 변환

        # gather를 이용해 정답 클래스에 해당하는 log_pt, pt만 추출
        # targets.unsqueeze(1)의 shape: (N, 1, H, W)
        log_pt = log_pt.gather(dim=1, index=targets.unsqueeze(1))  # (N, 1, H, W)
        log_pt = log_pt.view(-1)  # (N*H*W,)

        pt = pt.gather(dim=1, index=targets.unsqueeze(1))          # (N, 1, H, W)
        pt = pt.view(-1)                                           # (N*H*W,)

        # ignore_index가 설정되어 있으면, 해당 인덱스 위치는 무시
        if self.ignore_index is not None:
            valid_mask = (targets.view(-1) != self.ignore_index)
            log_pt = log_pt[valid_mask]
            pt = pt[valid_mask]

        # Focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma

        # alpha 적용 (특정 클래스에 가중치가 있는 경우, 여기서 클래스별 처리 가능)
        # 이 예시에선 alpha가 스칼라라고 가정
        loss = -self.alpha * focal_term * log_pt  # 음수 부호에 주의: -log_pt

        # reduction 방식 적용
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            # 'none'인 경우, (N*H*W,) 형태의 텐서를 그대로 반환
            return loss

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