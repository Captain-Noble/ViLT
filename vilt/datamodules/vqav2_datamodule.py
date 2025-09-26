from vilt.datasets import VQAv2Dataset
from .datamodule_base import BaseDataModule
from collections import defaultdict

class VQAv2DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VQAv2Dataset

    @property
    def dataset_name(self):
        return "vqa"

    def setup(self, stage):
        """
        保持原有设置（answer2id/id2answer），即便生成式不依赖它们；
        这样不影响旧评测/日志逻辑。
        """
        super().setup(stage)

        # 训练/验证集合答案表
        if self.train_dataset is not None and self.val_dataset is not None:
            train_answers = self.train_dataset.table["answers"].to_pandas().tolist()
            val_answers = self.val_dataset.table["answers"].to_pandas().tolist()
            train_labels = self.train_dataset.table["answer_labels"].to_pandas().tolist()
            val_labels = self.val_dataset.table["answer_labels"].to_pandas().tolist()

            all_answers = [c for c in train_answers + val_answers if c is not None]
            all_answers = [l for lll in all_answers for ll in lll for l in ll]
            all_labels = [c for c in train_labels + val_labels if c is not None]
            all_labels = [l for lll in all_labels for ll in lll for l in ll]

            self.answer2id = {k: v for k, v in zip(all_answers, all_labels)}
            sorted_a2i = sorted(self.answer2id.items(), key=lambda x: x[1])
            self.num_class = max(self.answer2id.values()) + 1 if len(self.answer2id) > 0 else 0
            self.id2answer = defaultdict(lambda: "unknown")
            for k, v in sorted_a2i:
                self.id2answer[v] = k
