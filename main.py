import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from project.utils.callbacks import ImageSegmentationLogger, checkpoint_callback, early_stop_callback


@hydra.main(config_path='config', config_name='defaults', version_base='1.1')
def main(cfg: DictConfig) -> None:

    pl.seed_everything(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Running on: " + str(device))

    logger = hydra.utils.instantiate(cfg.logger)

    model = hydra.utils.instantiate(cfg.model)

    pl_module = hydra.utils.instantiate(
        cfg.lightning_module,
        model=model,
        # Don't instantiate optimizer submodules with hydra, let `configure_optimizers()` do it
        _recursive_=False,
    )

    data_module = hydra.utils.instantiate(cfg.data)

    data_module.setup()
    val_samples = next(iter(data_module.val_dataloader()))

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        accelerator=str(device),
        logger=logger,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            ImageSegmentationLogger(val_samples)
        ]
    )

    trainer.fit(pl_module, data_module)


if __name__ == '__main__':
    main()
