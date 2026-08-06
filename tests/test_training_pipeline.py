import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from contextlib import redirect_stdout
import io
import sys

import torch
from torch.utils.data import DataLoader, Dataset

import aggregate_5fold_results as aggregate
import trainer


class ToyDataset(Dataset):
    def __init__(self, image_dir, label_dir, samples, **kwargs):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {'image': torch.ones(1, 4, 4), 'label': torch.ones(4, 4, dtype=torch.long)}


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.lists = self.directory / 'lists'
        self.lists.mkdir()
        for fold in (0, 1, 3, 4):
            (self.lists / f'fold{fold}.txt').write_text(f'case{fold}_slice0\n')
        self.snapshot = self.directory / 'model'
        self.snapshot.mkdir()
        self.args = SimpleNamespace(
            base_lr=0.01, batch_size=2, img_size=4, hu_stats_path='unused',
            root_path_5fold=str(self.directory), list_dir_5fold=str(self.lists),
            fold_idx=2, num_slices=1, seed=42, max_epochs=3,
            supervision='last_layer', dice_weight=0.5, ce_weight=0.5,
            training_loss_pilot=False, training_loss_patience=1, training_loss_min_delta=1e6,
        )

    def tearDown(self):
        for handler in list(logging.getLogger().handlers):
            logging.getLogger().removeHandler(handler)
            handler.close()
        self.temporary.cleanup()

    def run_training(self):
        model = torch.nn.Conv2d(1, 5, 1)

        def cpu_loader(dataset, **kwargs):
            kwargs['num_workers'] = 0
            kwargs['pin_memory'] = False
            return DataLoader(dataset, **kwargs)

        with (
            patch.object(trainer, 'COCAVolumeDataset', ToyDataset),
            patch.object(trainer, 'load_hu_stats', return_value={}),
            patch.object(trainer, 'DataLoader', side_effect=cpu_loader),
            patch.object(trainer, 'SummaryWriter', return_value=MagicMock()),
            patch.object(torch.Tensor, 'cuda', lambda tensor: tensor),
            patch.object(torch.cuda, 'device_count', return_value=0),
        ):
            trainer.trainer_coca(self.args, model, str(self.snapshot))
        return model

    def test_final_training_never_reads_test_fold_and_saves_last_weights(self):
        model = self.run_training()
        record = json.loads((self.snapshot / 'training_record.json').read_text())
        self.assertEqual(record['completed_epochs'], 3)
        self.assertEqual(record['training_folds'], [0, 1, 3, 4])
        self.assertEqual(record['selected_epochs'], 3)
        saved = torch.load(self.snapshot / 'final_model.pth', weights_only=True)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, saved[name]))
        self.assertEqual(len(list(self.snapshot.glob('*.pth'))), 1)

    def test_pilot_uses_training_loss_and_produces_no_evaluation_checkpoint(self):
        self.args.training_loss_pilot = True
        self.run_training()
        record = json.loads((self.snapshot / 'training_record.json').read_text())
        self.assertEqual(record['mode'], 'training_loss_pilot')
        self.assertEqual(record['completed_epochs'], 2)
        self.assertEqual(record['selected_epochs'], 1)
        self.assertFalse(list(self.snapshot.glob('*.pth')))

    def test_existing_checkpoint_is_preserved(self):
        old = self.snapshot / 'epoch_12_0.3_best_model.pth'
        old.write_bytes(b'previous run')
        with self.assertRaises(FileExistsError):
            self.run_training()
        self.assertEqual(old.read_bytes(), b'previous run')

    def test_aggregate_does_not_reuse_previous_or_incomplete_evaluations(self):
        log = self.directory / 'results.txt'
        lines = ['Evaluation checkpoint: /model/final_model.pth']
        lines.extend(f'[3D] Class {k} - Dice: 0.8, mIoU: 0.7, HD: 1.0' for k in range(1, 5))
        lines.append('[3D] Testing Performance - Mean Dice: 0.8, Mean mIoU: 0.7, Mean HD: 1.0')
        complete = '\n'.join(lines) + '\n'
        log.write_text(complete)
        self.assertEqual(aggregate.parse_results(log)['Dice'], [0.8] * 5)
        log.write_text(complete + 'Evaluation checkpoint: /other/final_model.pth\n')
        self.assertIsNone(aggregate.parse_results(log))
        log.write_text('\n'.join(lines[1:]))
        self.assertIsNone(aggregate.parse_results(log))

    def test_aggregate_resolves_different_epoch_counts_per_fold(self):
        model_root = self.directory / 'models'
        log_root = self.directory / 'logs'
        for fold, epochs in enumerate([1, 2, 3, 4, 5]):
            relative = Path('Unet_resnet50_sa/COCA_512') / f'run_fold{fold}' / f'epo{epochs}_bs16_lr1e-05'
            model_dir = model_root / relative
            model_dir.mkdir(parents=True)
            final = model_dir / 'final_model.pth'
            final.touch()
            record = {'mode': 'fixed_epochs', 'fold_idx': fold, 'training_folds': [k for k in range(5) if k != fold], 'selected_epochs': epochs, 'completed_epochs': epochs}
            (model_dir / 'training_record.json').write_text(json.dumps(record))
            log_dir = log_root / relative
            log_dir.mkdir(parents=True)
            lines = [f'Evaluation checkpoint: {final}']
            lines.extend(f'[3D] Class {k} - Dice: 0.8, mIoU: 0.7, HD: 1.0' for k in range(1, 5))
            lines.append('[3D] Testing Performance - Mean Dice: 0.8, Mean mIoU: 0.7, Mean HD: 1.0')
            (log_dir / 'results.txt').write_text('\n'.join(lines))
        output = self.directory / 'aggregate.md'
        arguments = ['aggregate', '--exp_template', 'run_fold{fold}', '--epochs_per_fold', '1', '2', '3', '4', '5',
                     '--model_root', str(model_root), '--test_log_root', str(log_root), '--out', str(output)]
        capture = io.StringIO()
        with patch.object(sys, 'argv', arguments), redirect_stdout(capture):
            aggregate.main()
        self.assertNotIn('[warn]', capture.getvalue())
        rendered = output.read_text()
        for fold, epochs in enumerate([1, 2, 3, 4, 5]):
            self.assertIn(f'| {fold} | {epochs} | {epochs} | final_model.pth |', rendered)
        self.assertIn('0.8000 ± 0.0000', rendered)


if __name__ == '__main__':
    unittest.main()
