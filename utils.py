import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

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
        """
        alpha: 스칼라 혹은 길이 C인 리스트/텐서
               - 예) alpha = [1.0, 2.0, 0.5, ...] 처럼 클래스별로 부여 가능
        gamma: Focal Loss의 (1 - pt)^gamma 에서의 gamma
        reduction: 'mean' | 'sum' | 'none'
        ignore_index: 무시할 클래스 인덱스 (ex. 배경)
        """
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (list, tuple)):
            # 리스트나 튜플이면 텐서로 변환
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs: (N, C, H, W) 형태의 로짓(logits)
        targets: (N, H, W) 형태로, 각 픽셀이 [0, C-1] 범위를 갖는 정수 레이블
        """
        # 1) log_softmax 계산
        log_pt = F.log_softmax(inputs, dim=1)  # (N, C, H, W)
        pt = torch.exp(log_pt)                 # (N, C, H, W)

        # 2) gather를 통해 각 픽셀의 정답 클래스 위치 인덱스에 해당하는 log_pt, pt만 추출
        log_pt = log_pt.gather(dim=1, index=targets.unsqueeze(1))  # (N, 1, H, W)
        pt = pt.gather(dim=1, index=targets.unsqueeze(1))          # (N, 1, H, W)

        # 3) 벡터화 위해 shape 조정 (N*H*W,)
        log_pt = log_pt.view(-1)
        pt = pt.view(-1)
        targets_flat = targets.view(-1)

        # 4) ignore_index 처리
        if self.ignore_index is not None:
            valid_mask = (targets_flat != self.ignore_index)
            log_pt = log_pt[valid_mask]
            pt = pt[valid_mask]
            targets_flat = targets_flat[valid_mask]

        # 5) focal_term 계산: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma

        # 6) alpha 적용
        #    - alpha가 스칼라인 경우: 모든 클래스에 동일하게 alpha
        #    - alpha가 길이 C인 벡터인 경우: targets_flat을 index로 해서 해당 alpha만 가져옴
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets_flat]  # (N*H*W,) 중 valid_mask 부분
        else:
            alpha_t = self.alpha  # 스칼라

        # 7) Focal Loss 최종 계산
        loss = -alpha_t * focal_term * log_pt  # -log_pt에 alpha와 focal_term 곱

        # 8) reduction 처리
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            # 'none'인 경우 (N*H*W,) 형태의 텐서 그대로 반환
            return loss
        
def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item