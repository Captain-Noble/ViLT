import os

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


import copy
import pytorch_lightning as pl

from vilt.config import ex
from vilt.modules import ViLTransformerSS
from vilt.datamodules.multitask_datamodule import MTDataModule
from pytorch_lightning.strategies import DDPStrategy



@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

    model = ViLTransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else 25000
    max_epochs = _config["max_epoch"] if _config["max_epoch"] is not None else 50
    # trainer = pl.Trainer(
    #     gpus=_config["num_gpus"],
    #     num_nodes=_config["num_nodes"],
    #     precision=_config["precision"],
    #     accelerator="ddp",
    #     benchmark=True,
    #     deterministic=True,
    #     max_epochs=_config["max_epoch"] if max_steps is None else 1000,
    #     max_steps=max_steps,
    #     callbacks=callbacks,
    #     logger=logger,
    #     prepare_data_per_node=False,
    #     replace_sampler_ddp=False,
    #     accumulate_grad_batches=grad_steps,
    #     log_every_n_steps=10,
    #     flush_logs_every_n_steps=10,
    #     resume_from_checkpoint=_config["resume_from"],
    #     weights_summary="top",
    #     fast_dev_run=_config["fast_dev_run"],
    #     val_check_interval=_config["val_check_interval"],
    # )

    trainer = pl.Trainer(
        # ▶️ 硬件/并行策略
        accelerator="gpu",                   # 旧写法 accelerator="ddp" → 现在写到 strategy
        # strategy="ddp",                      # 新增：并行策略放到 strategy
        
        strategy=DDPStrategy(find_unused_parameters=True),
        devices=_config["num_gpus"],         # 旧 gpus → 新 devices
        num_nodes=_config["num_nodes"],

        # ▶️ 训练精度与性能
        precision=_config["precision"],
        benchmark=True,
        deterministic=True,

        # ▶️ 训练时长控制
        max_epochs=max_epochs if max_steps is None else 1000,
        max_steps=max_steps,

        # ▶️ 记录与回调
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,

        # ▶️ 分布式采样器
        use_distributed_sampler=False,       # 旧 replace_sampler_ddp=False

        # ▶️ 梯度累积
        accumulate_grad_batches=grad_steps,

        # ▶️ 其他常用开关
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        # ← 从配置读取
        limit_train_batches=_config.get("limit_train_batches", 1.0),
        limit_val_batches=_config.get("limit_val_batches", 1.0),
        limit_test_batches=_config.get("limit_test_batches", 1.0),

    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
