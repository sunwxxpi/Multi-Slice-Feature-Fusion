import json
import tempfile
import unittest
from pathlib import Path

from training_protocol import TrainingLossPlateau, load_final_training_record


class TrainingProtocolTests(unittest.TestCase):
    def test_plateau_selects_training_loss_epoch(self):
        plateau = TrainingLossPlateau(patience=2, min_delta=0.01)
        self.assertFalse(plateau.observe(1, 0.8))
        self.assertFalse(plateau.observe(2, 0.6))
        self.assertFalse(plateau.observe(3, 0.595))
        self.assertTrue(plateau.observe(4, 0.7))
        self.assertEqual(plateau.best_epoch, 2)

    def test_new_improvement_resets_patience(self):
        plateau = TrainingLossPlateau(patience=2)
        for epoch, loss in enumerate([0.8, 0.9, 0.7, 0.8], start=1):
            self.assertFalse(plateau.observe(epoch, loss))
        self.assertTrue(plateau.observe(5, 0.9))
        self.assertEqual(plateau.best_epoch, 3)

    def test_nonfinite_pilot_loss_is_rejected(self):
        for loss in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                TrainingLossPlateau(2).observe(1, loss)

    def test_only_completed_matching_final_runs_are_evaluable(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            record = {'mode': 'fixed_epochs', 'fold_idx': 2, 'training_folds': [0, 1, 3, 4], 'selected_epochs': 3, 'completed_epochs': 3}
            path = directory / 'training_record.json'
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                load_final_training_record(directory, 2, 3)
            (directory / 'final_model.pth').touch()
            self.assertEqual(load_final_training_record(directory, 2, 3), record)
            for updates in [{'mode': 'training_loss_pilot'}, {'completed_epochs': 2}, {'fold_idx': 1}, {'selected_epochs': 4}, {'training_folds': [0, 1, 2, 3]}]:
                path.write_text(json.dumps(record | updates))
                with self.assertRaises(ValueError):
                    load_final_training_record(directory, 2, 3)


if __name__ == '__main__':
    unittest.main()
