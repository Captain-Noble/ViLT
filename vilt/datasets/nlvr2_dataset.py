from .base_dataset import BaseDataset
import sys
import random
import torch

class NLVR2Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["nlvr2_train"]
        elif split == "val":
            names = ["nlvr2_dev", "nlvr2_test1"]
        elif split == "test":
            names = ["nlvr2_dev", "nlvr2_test1"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        result = None
        while result is None:
            try:
                image_tensor_0 = self.get_image(index, image_key="image_0")["image"]
                image_tensor_1 = self.get_image(index, image_key="image_1")["image"]
                text = self.get_text(index)["text"]
                result = True
            except Exception:
                print(
                    f"error while read file idx {index} in {self.names[0]}",
                    file=sys.stderr,
                )
                index = random.randint(0, len(self.index_mapper) - 1)

        index, question_index = self.index_mapper[index]
        ans_str = self.table["answers"][index][question_index].as_py()
        answers_bool = (ans_str == "True")

        # 生成式文本
        gen_input_text  = f"question: {text} answer:"
        gen_target_text = "true" if answers_bool else "false"

        return {
            "image_0": image_tensor_0,
            "image_1": image_tensor_1,
            "text": text,                    # 兼容原本管线
            "answers": answers_bool,         # 兼容原本的分类 fine-tune
            "table_name": self.table_names[index],
            # 生成式新增
            "gen_input_text": gen_input_text,
            "gen_target_text": gen_target_text,
        }

    # 覆写 collate：在父类的基础上，额外编码 gen_input/gen_target
    def collate(self, batch, mlm_collator=None):
        ret = super().collate(batch, mlm_collator=mlm_collator)

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

            ret["gen_input_ids"]    = tok_inp["input_ids"]
            ret["gen_input_masks"]  = tok_inp["attention_mask"]
            ret["gen_target_ids"]   = tok_tgt["input_ids"]
            ret["gen_target_masks"] = tok_tgt["attention_mask"]
        else:
            ret["gen_input_ids"]    = torch.empty(0, dtype=torch.long)
            ret["gen_input_masks"]  = torch.empty(0, dtype=torch.long)
            ret["gen_target_ids"]   = torch.empty(0, dtype=torch.long)
            ret["gen_target_masks"] = torch.empty(0, dtype=torch.long)

        return ret
