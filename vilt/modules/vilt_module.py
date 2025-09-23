import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        # for newer version
                # ▶️ 用于跨 batch 缓存各阶段 step 输出
        self.test_step_outputs: list[dict] = []        # test
        self.training_step_outputs: list[torch.Tensor] = []  # train
        self.validation_step_outputs: list = []        # val
        # ---------------

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)


        # ✅ 提前定义 hs，后面 ProjectionHead 要用
        hs = self.hparams.config["hidden_size"]
        # ===== 预训练任务头 =====
        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # === 新增：AR-LM 头 ===
        if config["loss_names"].get("ar_lm", 0) > 0:
            tie = self.hparams.config.get("tie_lm_head", True)
            weight = (
                self.text_embeddings.word_embeddings.weight
                if tie else None
            )
            self.lm_score = heads.LMHead(
                hidden_size=config["hidden_size"],
                vocab_size=config["vocab_size"],
                weight=weight,
                bias=True,
            )
            if not tie:
                self.lm_score.apply(objectives.init_weights)
        # === ITC 投影与温度（新增） ===
        if self.hparams.config["loss_names"].get("itc", 0) > 0:
            proj_dim = int(self.hparams.config.get("proj_dim", hs))
            ratio = self.hparams.config.get("proj_mlp_ratio", 1.0)
            drop = self.hparams.config.get("proj_dropout", 0.0)
            self.text_proj  = heads.ProjectionHead(hs, proj_dim, hidden_dim=int(hs*ratio), dropout=drop)
            self.image_proj = heads.ProjectionHead(hs, proj_dim, hidden_dim=int(hs*ratio), dropout=drop)

            self.text_proj.apply(objectives.init_weights)
            self.image_proj.apply(objectives.init_weights)
            # CLIP风格的可学习温度（logit_scale存的是log温度）
            import math, torch
            init = float(self.hparams.config.get("logit_scale_init", 1/0.07))
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(init)))

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu", weights_only=False)
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            # 可能不再有 itm_score，因此要做防御
            if hasattr(self, "itm_score"):
                self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
                self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
                for p in self.itm_score.parameters():
                    p.requires_grad = False
            self.margin = 0.2

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu", weights_only=False)
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
                causal_lm: bool = False,   # ← 新增
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )
        

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds
        # ===== 新增：构造因果注意力的 pairwise mask（B,1,L,L） =====
        pair_mask = None
        if causal_lm and self.hparams.config.get("ar_causal_mask", True):
            B, Lt = text_masks.shape
            Li = image_masks.shape[1]
            L  = Lt + Li
            device = x.device

            # 文本-文本：严格下三角（含自身）
            tri = torch.tril(torch.ones(Lt, Lt, dtype=torch.bool, device=device))
            tri = tri.unsqueeze(0).unsqueeze(1).expand(B, 1, Lt, Lt)  # [B,1,Lt,Lt]
            # 文本-图像：全可见
            ones_ti = torch.ones(B, 1, Lt, Li, dtype=torch.bool, device=device)
            attn_text = torch.cat([tri, ones_ti], dim=3)             # [B,1,Lt,L]

            # 图像 query：不做限制（全可见）
            attn_img = torch.ones(B, 1, Li, L, dtype=torch.bool, device=device)

            pair_mask = torch.cat([attn_text, attn_img], dim=2)      # [B,1,L,L]
            # 结合 key 的有效性（把 padding key 屏蔽掉）
            pair_mask = pair_mask & co_masks[:, None, None, :].bool()

        # 传入 Attention：支持 (B,L) 或 (B,1,L,L) 两种形式
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=(pair_mask if pair_mask is not None else co_masks))

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }
        return ret


    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))
        # === CLIP式对比对齐（新增） ===
        if "itc" in self.current_tasks:
            ret.update(objectives.compute_itc(self, batch))

        # === 新增：AR-LM ===
        if "ar_lm" in self.current_tasks:
            ret.update(objectives.compute_ar_lm(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        self.training_step_outputs.append(total_loss.detach())
        return total_loss
    
    # def training_epoch_end(self, outs):
        # vilt_utils.epoch_wrapup(self)
    # 旧：training_epoch_end → 新：on_train_epoch_end
    def on_train_epoch_end(self):
        vilt_utils.epoch_wrapup(self)
        self.training_step_outputs.clear()  # 释放显存

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        self.validation_step_outputs.append(output)

    # def validation_epoch_end(self, outs):
    #     vilt_utils.epoch_wrapup(self)
        # 旧：validation_epoch_end → 新：on_validation_epoch_end
    def on_validation_epoch_end(self):
        vilt_utils.epoch_wrapup(self)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))


        #  for newer version
        # 记录当前 batch 的输出，供 epoch_end 聚合
        self.test_step_outputs.append(ret)
        # ---------------

        return ret

    # def test_epoch_end(self, outs):
    #     model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

    #     if self.hparams.config["loss_names"]["vqa"] > 0:
    #         objectives.vqa_test_wrapup(outs, model_name)
    #     vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)

    def on_test_epoch_end(self):
        outs = self.test_step_outputs  

        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

        self.test_step_outputs.clear()
    # === 放在 ViLTransformerSS 类内部 ===

    def _debug_is_on(self) -> bool:
        # 在 config 里加个开关：debug_unused = True/False（默认 False）
        return bool(self.hparams.config.get("debug_unused", False))

    def _group_name(self, full_name: str) -> str:
        # 取参数名前缀，方便按头聚合
        return full_name.split(".", 1)[0] if "." in full_name else full_name

    def _print_once(self, msg: str):
        # 只在 rank0 打印，避免多卡刷屏
        try:
            is_main = (self.trainer is None) or self.trainer.is_global_zero
        except Exception:
            is_main = True
        if is_main:
            print(msg, flush=True)

    def on_train_batch_start(self, batch, batch_idx):
        # 训练前，看看这步跑哪些任务（vilt_utils.set_task 已经在 training_step 开头调用）
        if self._debug_is_on():
            self._print_once(f"[step {getattr(self, 'global_step', -1)}] current_tasks={self.current_tasks}")

    def on_after_backward(self):
        """
        每个训练 step 反传后调用：统计这一步哪些参数 grad 仍是 None → 没参与本次 loss。
        """
        if not self._debug_is_on():
            return

        unused = {}
        used_cnt = 0
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                grp = self._group_name(name)
                unused.setdefault(grp, []).append(name)
            else:
                used_cnt += 1

        if not unused:
            self._print_once(f"[step {getattr(self, 'global_step', -1)}] All trainable params used in this step. (used_cnt={used_cnt})")
            return

        # 只打印每个 group 的数量和前几项，避免刷屏
        summary_lines = [f"[step {getattr(self, 'global_step', -1)}] Unused parameters this step (by group):"]
        for grp, names in sorted(unused.items()):
            preview = ", ".join(names[:3]) + (" ..." if len(names) > 3 else "")
            summary_lines.append(f"  - {grp:<18} count={len(names):<5} e.g. {preview}")
        self._print_once("\n".join(summary_lines))
