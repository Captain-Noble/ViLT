import torch
import random

# from transformers.optimization import AdamW
from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from vilt.modules.dist_utils import all_gather
from vilt.modules.objectives import compute_irtr_recall
from vilt.gadgets.my_metrics import Accuracy, VQAScore, Scalar


def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:
                continue
            if k == "vqa":
                setattr(pl_module, f"{split}_vqa_score", VQAScore())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "nlvr2":
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"dev_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
            elif k == "irtr":
                setattr(pl_module, f"{split}_irtr_loss", Scalar())
            elif k == "mppd" or k == "mpfr":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "itm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_wpa_loss", Scalar())
            elif k in ["vqa_gen", "nlvr2_gen", "imgcls_gen"]:
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())


def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0

    if pl_module.hparams.config["get_recall_metric"] and not pl_module.training:
        (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall(pl_module)
        print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10), pl_module.global_step)
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r1", ir_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r5", ir_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r10", ir_r10, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r1", tr_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r5", tr_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r10", tr_r10, pl_module.global_step
        )
        the_metric += ir_r1.item() + tr_r1.item()

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue

        value = 0

        if loss_name == "vqa":
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "nlvr2":
            if phase == "train":
                value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/train/accuracy_epoch", value)
                getattr(pl_module, f"train_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/train/loss_epoch",
                    getattr(pl_module, f"train_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"train_{loss_name}_loss").reset()
            else:
                value = getattr(pl_module, f"dev_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/dev/accuracy_epoch", value)
                getattr(pl_module, f"dev_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/dev/loss_epoch",
                    getattr(pl_module, f"dev_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"dev_{loss_name}_loss").reset()

                value = getattr(pl_module, f"test_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/test/accuracy_epoch", value)
                getattr(pl_module, f"test_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/test/loss_epoch",
                    getattr(pl_module, f"test_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"test_{loss_name}_loss").reset()
        elif loss_name == "irtr":
            pl_module.log(
                f"{loss_name}/{phase}/irtr_loss_epoch",
                getattr(pl_module, f"{phase}_irtr_loss").compute(),
            )
            getattr(pl_module, f"{phase}_irtr_loss").reset()
        elif loss_name == "mppd" or loss_name == "mpfr":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "itm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            pl_module.log(
                f"{loss_name}/{phase}/wpa_loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").reset()
        # added part for generative finetune
        elif loss_name in ["vqa_gen", "nlvr2_gen", "imgcls_gen"]:
            # 记录 loss
            gen_loss = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            pl_module.log(f"{loss_name}/{phase}/loss_epoch", gen_loss, sync_dist=True)

            # 用 token-level accuracy 或 ppl 的倒数做 the_metric（>=0 且能继续变大）
            if hasattr(pl_module, f"{phase}_{loss_name}_accuracy"):
                gen_acc = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", gen_acc, sync_dist=True)
                value = gen_acc
                getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            else:
                # 若没做 meter，就用 1/(1+loss) 作为正向指标
                value = 1.0 / (1.0 + gen_loss)

            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric)


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
    ]
    return


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias", "LayerNorm.bias", "LayerNorm.weight",
        "norm.bias", "norm.weight", "norm1.bias", "norm1.weight",
        "norm2.bias", "norm2.weight",
    ]
    head_names = ["vqa_classifier", "nlvr2_classifier", "lm_score", "text_proj", "image_proj"]
    lr_mult = pl_module.hparams.config["lr_mult"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    # ——— 找到“共享词表”的 Parameter 对象（若未绑则为 None）———
    tied_vocab_param = None
    if hasattr(pl_module, "lm_score") and hasattr(pl_module.lm_score, "decoder"):
        try:
            tied_vocab_param = pl_module.lm_score.decoder.weight
        except Exception:
            tied_vocab_param = None

    def is_tied_vocab(n, p):
        return (tied_vocab_param is not None) and (p is tied_vocab_param)

    # ——— 分组 —— 注意：①② 里都显式排除 is_tied_vocab(n,p) ———
    optimizer_grouped_parameters = [
        {   # (1) backbone & 有权重衰减
            "params": [
                p for n, p in pl_module.named_parameters()
                if (not any(nd in n for nd in no_decay))
                and (not any(bb in n for bb in head_names))
                and (not is_tied_vocab(n, p))              # ① 排除共享词表
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {   # (2) backbone & 无权重衰减
            "params": [
                p for n, p in pl_module.named_parameters()
                if (any(nd in n for nd in no_decay))
                and (not any(bb in n for bb in head_names))
                and (not is_tied_vocab(n, p))              # ② 排除共享词表
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {   # (3) heads & 有权重衰减（包含共享词表）
            "params": [
                p for n, p in pl_module.named_parameters()
                if (not any(nd in n for nd in no_decay))
                and ( any(bb in n for bb in head_names) or is_tied_vocab(n, p) )
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {   # (4) heads & 无权重衰减（包含共享词表）
            "params": [
                p for n, p in pl_module.named_parameters()
                if (any(nd in n for nd in no_decay))
                and ( any(bb in n for bb in head_names) or is_tied_vocab(n, p) )
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    # —— 可选：启动期做一次重复参数断言（方便快速定位问题）——
    seen = set()
    for gi, g in enumerate(optimizer_grouped_parameters):
        for p in g["params"]:
            pid = id(p)
            if pid in seen:
                raise RuntimeError(f"Parameter appears in multiple groups (group idx={gi}).")
            seen.add(pid)

    # === 优化器与调度 ===
    if optim_type == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps is None:
        max_steps = (len(pl_module.trainer.datamodule.train_dataloader())
                     * pl_module.trainer.max_epochs
                     // pl_module.trainer.accumulate_grad_batches)
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(warmup_steps, float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer, warmup_steps, max_steps, lr_end=end_lr, power=decay_power
        )
    return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
