from .base_dataset import BaseDataset
import torch

class VQAv2Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["vqav2_train", "vqav2_trainable_val"]
        elif split == "val":
            names = ["vqav2_rest_val"]
        elif split == "test":
            names = ["vqav2_test"]  # vqav2_test-dev for test-dev

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        if self.split != "test":
            answers = self.table["answers"][index][question_index].as_py()
            labels = self.table["answer_labels"][index][question_index].as_py()
            scores = self.table["answer_scores"][index][question_index].as_py()
            # 生成式：选一个最高分答案作为 target 文本
            if scores and answers:
                # 取分数最大的那个答案
                best_i = max(range(len(scores)), key=lambda i: scores[i])
                target_text = str(answers[best_i])
            elif answers:
                target_text = str(answers[0])
            else:
                target_text = ""
        else:
            answers, labels, scores = [], [], []
            target_text = ""  # 测试时没有标签，target 给空串

        gen_input_text = f"question: {text} answer:"
        gen_target_text = target_text

        return {
            "image": image_tensor,
            "text": text,                       # 兼容原本管线（如 mlm/itm）
            "vqa_answer": answers,              # 兼容老评估
            "vqa_labels": labels,
            "vqa_scores": scores,
            "qid": qid,
            # 生成式新增
            "gen_input_text": gen_input_text,
            "gen_target_text": gen_target_text,
        }

    # 覆写 collate：在父类的基础上，额外编码 gen_input/gen_target
    def collate(self, batch, mlm_collator=None):
        ret = super().collate(batch, mlm_collator=mlm_collator)

        # 允许 batch 里有别的数据源样本 → 用 get 防御
        gen_inputs  = [b.get("gen_input_text", "") for b in batch]
        gen_targets = [b.get("gen_target_text", "") for b in batch]

        if hasattr(self, "tokenizer") and len(gen_inputs) > 0:
            max_len = getattr(self, "max_text_len", 40)

            tok_inp = self.tokenizer(
                gen_inputs,
                padding="longest",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
                add_special_tokens=True,
            )
            tok_tgt = self.tokenizer(
                gen_targets,
                padding="longest",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
                add_special_tokens=True,
            )

            ret["gen_input_ids"]   = tok_inp["input_ids"]
            ret["gen_input_masks"] = tok_inp["attention_mask"]
            ret["gen_target_ids"]  = tok_tgt["input_ids"]
            ret["gen_target_masks"]= tok_tgt["attention_mask"]
        else:
            # 兜底，避免 key 不存在
            ret["gen_input_ids"]    = torch.empty(0, dtype=torch.long)
            ret["gen_input_masks"]  = torch.empty(0, dtype=torch.long)
            ret["gen_target_ids"]   = torch.empty(0, dtype=torch.long)
            ret["gen_target_masks"] = torch.empty(0, dtype=torch.long)

        return ret