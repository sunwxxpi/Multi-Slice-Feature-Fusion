import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingLossPlateau:
    patience: int
    min_delta: float = 0.0
    best_loss: float = math.inf
    best_epoch: int = 0
    epochs_without_improvement: int = 0

    def observe(self, epoch, loss):
        """학습 loss의 개선 여부를 기록하고 종료 조건을 반환한다."""
        if not math.isfinite(loss):
            raise ValueError('학습 loss가 유한하지 않습니다.')
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


def load_final_training_record(model_dir, fold_idx, max_epochs):
    """완료된 지정 epoch 학습의 기록을 반환한다."""
    directory = Path(model_dir)
    with (directory / 'training_record.json').open() as stream:
        record = json.load(stream)
    if (
        not isinstance(record, dict)
        or record.get('mode') != 'fixed_epochs'
        or record.get('fold_idx') != fold_idx
        or record.get('training_folds') != [k for k in range(5) if k != fold_idx]
        or record.get('selected_epochs') != max_epochs
        or record.get('completed_epochs') != max_epochs
        or not (directory / 'final_model.pth').is_file()
    ):
        raise ValueError('평가 fold 또는 epoch 설정과 완료된 최종 학습 기록이 일치하지 않습니다.')
    return record
