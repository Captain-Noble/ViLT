import random
import torch
import io
import pyarrow as pa
import os

from PIL import Image
from vilt.transforms import keys_to_transforms


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int,
        names: list,
        text_column_name: str = "",
        remove_duplicate=True,
        max_text_len=40,
        draw_false_image=0,
        draw_false_text=0,
        image_only=False,
        max_num=None,   # 🔹 新增参数
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        max_num : if not None, limit number of loaded samples
        """
        assert len(transform_keys) >= 1
        # max_num = 256
        super().__init__()

        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.max_num = max_num   # 🔹 保存参数

        if len(names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]

            self.table_names = list()
            for i, name in enumerate(names):
                if i >= len(tables):
                    break
                self.table_names += [name] * len(tables[i])

            self.table = pa.concat_tables(tables, promote=True)

            # 🔹 如果 max_num 不为 None，裁剪 table
            if self.max_num is not None and len(self.table) > self.max_num:
                self.table = self.table.slice(0, self.max_num)
                self.table_names = self.table_names[: self.max_num]

            if text_column_name != "":
                self.text_column_name = text_column_name
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                self.all_texts = (
                    [list(set(texts)) for texts in self.all_texts]
                    if remove_duplicate
                    else self.all_texts
                )
            else:
                self.all_texts = list()
        else:
            self.all_texts = list()

        self.index_mapper = dict()

        if text_column_name != "" and not self.image_only:
            j = 0
            for i, texts in enumerate(self.all_texts):
                for _j in range(len(texts)):
                    self.index_mapper[j] = (i, _j)
                    j += 1
        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper)

    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {
            "image": image_tensor,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }

    def get_false_image(self, rep, image_key="image"):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        image = self.get_raw_image(random_index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {f"false_image_{rep}": image_tensor}

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]

        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.index_mapper) - 1)

        index, caption_index = self.index_mapper[random_index]
        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                if not self.image_only:
                    txt = self.get_text(index)
                    ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    ret.update(txt)

                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
        return ret
    
    def collate(self, batch, mlm_collator):
        """
        统一支持：
        - MLM 预训练（mlm_collator 不为 None）
        - 生成式/AR-LM（mlm_collator 为 None）
        产出键：
        <txt_key>                      : 原始文本列表
        <txt_key>_ids                 : [B, L] 未mask ids
        <txt_key>_masks               : [B, L] attention mask
        <txt_key>_labels              : [B, L] 全 -100（生成式监督在 objectives 中构造）
        仅 MLM 路径额外：
        <txt_key>_ids_mlm             : [B, Lm]
        <txt_key>_labels_mlm          : [B, Lm]
        """
        import torch

        # -------- helpers --------
        def _resolve_cap_len():
            cap = getattr(self, "max_text_len", None)
            if isinstance(cap, int) and cap > 0:
                return cap
            tmax = getattr(getattr(self, "tokenizer", None), "model_max_length", None)
            if isinstance(tmax, int) and 0 < tmax < 100000:
                return tmax
            return 40

        def _extract_text(x):
            """把样本里的 text 稳妥抽成 str。支持:
            - str
            - (text, encoding) / [text, encoding]
            - dict 含 'text' / 'raw_text'
            - 其它 -> str(x)
            """
            if x is None:
                return ""
            if isinstance(x, (tuple, list)) and len(x) > 0:
                x = x[0]
            elif isinstance(x, dict):
                if "text" in x:
                    x = x["text"]
                elif "raw_text" in x:
                    x = x["raw_text"]
            if x is None:
                return ""
            if isinstance(x, bytes):
                try:
                    x = x.decode("utf-8", errors="ignore")
                except Exception:
                    x = ""
            if not isinstance(x, str):
                x = str(x)
            return x

        cap_len = _resolve_cap_len()
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        # ===================== 图像对齐 =====================
        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = []
        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i if ii is not None]
        if len(img_sizes) > 0:
            for size in img_sizes:
                assert len(size) == 3, f"expect CHW, got {size}"
            max_height = max([i[1] for i in img_sizes])
            max_width  = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]
            base_tensor = None
            for i in img:
                if i is not None:
                    for v in i:
                        if v is not None:
                            base_tensor = v
                            break
                if base_tensor is not None:
                    break
            if base_tensor is None or len(img_sizes) == 0:
                continue
            # 兼容第一条是 None 的情况，找一个非 None 的样本决定 view_size
            if img[0] is not None:
                view_size = len(img[0])
            else:
                sample_nonnull = next((x for x in img if x is not None), None)
                view_size = len(sample_nonnull) if sample_nonnull is not None else 1

            new_images = [torch.zeros(batch_size, 3, max_height, max_width, dtype=base_tensor.dtype)
                        for _ in range(view_size)]
            for bi in range(batch_size):
                per_sample = img[bi]
                for vi in range(view_size):
                    if (per_sample is None) or (per_sample[vi] is None):
                        continue
                    orig = per_sample[vi]
                    new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig
            dict_batch[img_key] = new_images

        # ===================== 文本对齐 =====================
        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
        if len(txt_keys) != 0:
            # 先抽出纯字符串，再 tokenizer（不 padding，截断到 cap_len），后续我们自己对齐
            encodings_per_key = {}
            raw_texts_per_key = {}
            for txt_key in txt_keys:
                texts_i = [ _extract_text(t) for t in dict_batch[txt_key] ]
                raw_texts_per_key[txt_key] = texts_i
                enc_i = [
                    self.tokenizer(
                        t,
                        padding=False,
                        truncation=True,
                        max_length=cap_len,
                    )
                    for t in texts_i
                ]
                encodings_per_key[txt_key] = enc_i

            if mlm_collator is not None:
                # ---------- MLM 路径 ----------
                flatten = []
                for txt_key in txt_keys:
                    for e in encodings_per_key[txt_key]:
                        ids = torch.tensor(e["input_ids"], dtype=torch.long)
                        if ids.numel() > cap_len:
                            ids = ids[:cap_len]
                        flatten.append(ids)

                mlm_out = mlm_collator(flatten)  # {'input_ids','labels'}
                Lm = mlm_out["input_ids"].size(1)

                offset = 0
                for txt_key in txt_keys:
                    enc_i = encodings_per_key[txt_key]
                    max_len = min(max((len(e["input_ids"]) for e in enc_i), default=1), cap_len)
                    input_ids       = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
                    attention_masks = torch.zeros((batch_size, max_len), dtype=torch.long)
                    for bi, e in enumerate(enc_i):
                        ids = torch.tensor(e["input_ids"], dtype=torch.long)[:max_len]
                        am  = torch.tensor(e["attention_mask"], dtype=torch.long)[:max_len]
                        L   = ids.numel()
                        input_ids[bi, :L]       = ids
                        attention_masks[bi, :L] = am

                    mlm_ids    = mlm_out["input_ids"][offset:offset + batch_size]      # [B, Lm]
                    mlm_labels = mlm_out["labels"][offset:offset + batch_size]         # [B, Lm]
                    offset += batch_size

                    dict_batch[f"{txt_key}_ids"]        = input_ids
                    dict_batch[f"{txt_key}_masks"]      = attention_masks
                    dict_batch[f"{txt_key}_labels"]     = torch.full_like(input_ids, -100)
                    dict_batch[f"{txt_key}_ids_mlm"]    = mlm_ids
                    dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                    dict_batch[txt_key] = raw_texts_per_key[txt_key]

            else:
                # ---------- 生成式/AR-LM 路径 ----------
                for txt_key in txt_keys:
                    enc_i = encodings_per_key[txt_key]
                    max_len = min(max((len(e["input_ids"]) for e in enc_i), default=1), cap_len)
                    input_ids       = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
                    attention_masks = torch.zeros((batch_size, max_len), dtype=torch.long)
                    for bi, e in enumerate(enc_i):
                        ids = torch.tensor(e["input_ids"], dtype=torch.long)[:max_len]
                        am  = torch.tensor(e["attention_mask"], dtype=torch.long)[:max_len]
                        L   = ids.numel()
                        input_ids[bi, :L]       = ids
                        attention_masks[bi, :L] = am

                    dict_batch[f"{txt_key}_ids"]    = input_ids
                    dict_batch[f"{txt_key}_masks"]  = attention_masks
                    dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                    dict_batch[txt_key] = raw_texts_per_key[txt_key]

        return dict_batch
        