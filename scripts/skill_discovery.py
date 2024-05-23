import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import wandb
from xskill.dataset.dataset import ConcatDataset
from xskill.utility.transform import get_transform_pipeline
from lightning.pytorch import seed_everything


@hydra.main(version_base=None,
            config_path="../config/simulation/",
            config_name="skill_discovery")
def pretrain(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    print(f"output_dir: {output_dir}")
    pretrain_pipeline = get_transform_pipeline(cfg.augmentations)

    seed_everything(cfg.seed, workers=True)
    robot_dataset = hydra.utils.instantiate(cfg.robot_dataset)
    human_dataset = hydra.utils.instantiate(cfg.human_dataset)
    combine_dataset = ConcatDataset(robot_dataset, human_dataset)

    paired_dataset = hydra.utils.instantiate(cfg.paired_dataset)

    dataloader = torch.utils.data.DataLoader(
        combine_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        drop_last=cfg.drop_last)


    steps_per_epoch = len(dataloader)

    model = hydra.utils.instantiate(
        cfg.Model,
        steps_per_epoch=steps_per_epoch,
        pretrain_pipeline=pretrain_pipeline,
        paired_dataset=paired_dataset
    )

    print("dataset len: ", len(combine_dataset))
    print(combine_dataset[1][0].im_q.shape)

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=cfg.callback.every_n_epoch,
        save_top_k=-1,
        dirpath=output_dir,
        filename="{epoch:02d}",
    )

    # Set up logger
    wandb.init(project="kitchen_prototype_learning", name = f"{output_dir}")
    # wandb_logger = WandbLogger(project="visual_skill_prior")
    wandb.config.update(OmegaConf.to_container(cfg))
    trainer = pl.Trainer(
        # logger=wandb_logger,
        callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint_callback],
        enable_checkpointing=True,
        default_root_dir=output_dir,
        deterministic=True,
        **cfg.Trainer,
    )

    trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == "__main__":
    pretrain()
