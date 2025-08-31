import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def path2rest(path, iid2captions, iid2split):
    name = path.split("/")[-1]

    with open(path, "rb") as fp:
        binary = fp.read()

    captions = iid2captions[name]
    split = iid2split[name]

    return [binary, captions, name, split]


def make_arrow(root, dataset_root, num_limit: int = -1):
    RAND_SEED = 123

    with open(f"{root}/karpathy/dataset_flickr30k.json", "r") as fp:
        captions = json.load(fp)
    captions = captions["images"]

    iid2captions = defaultdict(list)
    iid2split = dict()

    for cap in tqdm(captions):
        filename = cap["filename"]
        iid2split[filename] = cap["split"]
        for c in cap["sentences"]:
            iid2captions[filename].append(c["raw"])

    paths = list(glob(f"{root}/flickr30k-images/*.jpg"))
    random.shuffle(paths)
    caption_paths = [p for p in paths if p.split("/")[-1] in iid2captions]

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(len(paths), len(caption_paths), len(iid2captions))

    # —— 采样（按 split）
    if num_limit is not None and num_limit > 0:
        random.seed(RAND_SEED)
        by_split = defaultdict(list)
        for p in caption_paths:
            by_split[iid2split[p.split('/')[-1]]].append(p)
        caption_paths = []
        for sp in ["train", "val", "test"]:
            lst = by_split.get(sp, [])
            k = min(num_limit, len(lst))
            if k > 0:
                caption_paths.extend(random.sample(lst, k))
        random.shuffle(caption_paths)
        print(f"[sampling] num_limit={num_limit} per split -> final={len(caption_paths)}")

    bs = [path2rest(p, iid2captions, iid2split) for p in tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/f30k_caption_karpathy_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
