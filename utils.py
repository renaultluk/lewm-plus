import numpy as np
import shutil
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback
from pathlib import Path

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, run_dir, cache_dir=None, resume_alias=None, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.cache_dir = cache_dir
        self.resume_alias = resume_alias
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save_pt(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save_pt(pl_module.model, trainer.current_epoch + 1)

            self._refresh_resume_alias()

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        if trainer.is_global_zero:
            self._refresh_resume_alias()

    def _save_pt(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
            cache_dir=self.cache_dir,
        )

    def _refresh_resume_alias(self):
        if not self.resume_alias:
            return

        last_ckpt = self.run_dir / "last.ckpt"
        if not last_ckpt.exists():
            return

        alias_path = self.run_dir / self.resume_alias
        shutil.copy2(last_ckpt, alias_path)
