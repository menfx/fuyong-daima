"""
可复用数据加载模板。

三种模式（通过 mode 参数切换）：
    "pretrain"       — 原始文本 → 滑动窗口 DataLoader（ch02 风格）
    "classification" — CSV (text + label) → padding DataLoader（ch06 风格）
    "instruction"    — JSON (instruction/input/output) → Alpaca 格式化 +
                        动态 batch padding DataLoader（ch07 风格）

用法：
    from templates.data_template import build_dataloader

    loader = build_dataloader("data.txt", mode="pretrain", batch_size=4)

定制点：搜索 "【定制区】"，按需求修改对应代码块。
"""

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


# ============================================================
# 核心入口
# ============================================================

def build_dataloader(source, mode="pretrain", tokenizer_name="gpt2",
                     batch_size=4, max_length=256, stride=128,
                     shuffle=True, drop_last=True, num_workers=0,
                     pad_token_id=50256, ignore_index=-100,
                     device="cpu", label_col="Label", text_col="Text"):
    """
    统一数据加载入口。

    参数：
        source          : 文件路径（.txt / .csv / .json）
        mode            : "pretrain" | "classification" | "instruction"
        tokenizer_name  : tiktoken 编码名，默认 "gpt2"
        batch_size      : 批次大小
        max_length      : 序列最大长度 / 上下文窗口
        stride          : (仅 pretrain) 滑动窗口步长
        shuffle         : 是否打乱
        drop_last       : 是否丢弃不完整 batch
        num_workers     : DataLoader 工作进程数
        pad_token_id    : 填充 token ID
        ignore_index    : (仅 instruction) target 中被忽略的填充值
        device          : (仅 instruction) 数据所在设备
        label_col       : (仅 classification) CSV 标签列名
        text_col        : (仅 classification) CSV 文本列名

    返回：
        DataLoader（pretrain / classification）
        或 (train_loader, val_loader, test_loader) — instruction 模式自动划分
    """
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    # 【定制区】用自定义 tokenizer 时改这行
    # tokenizer = your_custom_tokenizer

    if mode == "pretrain":
        return _build_pretrain(source, tokenizer, batch_size, max_length,
                               stride, shuffle, drop_last, num_workers)

    elif mode == "classification":
        return _build_classification(source, tokenizer, batch_size,
                                     max_length, shuffle, drop_last,
                                     num_workers, pad_token_id,
                                     label_col, text_col)

    elif mode == "instruction":
        return _build_instruction(source, tokenizer, batch_size, max_length,
                                  shuffle, drop_last, num_workers,
                                  pad_token_id, ignore_index, device)

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# 模式一：预训练（原始文本）
# ============================================================

def _build_pretrain(file_path, tokenizer, batch_size, max_length,
                    stride, shuffle, drop_last, num_workers):
    """原始文本 → 滑动窗口 DataLoader。"""
    # 【定制区】替换为你自己的文本加载逻辑
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    dataset = _PretrainDataset(text, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers,
                      pin_memory=True, persistent_workers=num_workers > 0,
                      prefetch_factor=2)


class _PretrainDataset(Dataset):
    """滑动窗口数据集。
    
    注意：当前实现在 __init__ 时将整个文本全部 tokenize 并存入内存。
    【定制区】大数据集（>1GB）时改为在 __getitem__ 中按需加载，
    或使用 IterableDataset 流式读取。
    """

    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1:i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# ============================================================
# 模式二：分类微调（CSV）
# ============================================================

def _build_classification(csv_path, tokenizer, batch_size, max_length,
                          shuffle, drop_last, num_workers, pad_token_id,
                          label_col, text_col):
    """CSV → padding DataLoader。"""
    import pandas as pd

    # 【定制区】替换为你自己的 CSV 加载逻辑
    df = pd.read_csv(csv_path)

    # 【定制区】类别不平衡时取消注释，自动欠采样
    # from llms_from_scratch.ch06 import create_balanced_dataset
    # df = create_balanced_dataset(df)

    dataset = _ClassificationDataset(df, tokenizer, max_length,
                                     pad_token_id, label_col, text_col)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers,
                      pin_memory=True, persistent_workers=num_workers > 0,
                      prefetch_factor=2)


class _ClassificationDataset(Dataset):
    """分类数据集 —— 自动 tokenize + padding 到统一长度。"""

    def __init__(self, df, tokenizer, max_length, pad_token_id,
                 label_col, text_col):
        self.labels = df[label_col].tolist()
        self.encoded_texts = []
        for text in df[text_col]:
            ids = tokenizer.encode(text)
            # 截断过长文本
            ids = ids[:max_length] if max_length else ids
            self.encoded_texts.append(ids)
        self.max_length = max_length or max(len(ids) for ids in self.encoded_texts)
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = self.encoded_texts[idx]
        # padding
        padded = ids + [self.pad_token_id] * (self.max_length - len(ids))
        return (torch.tensor(padded, dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))


# ============================================================
# 模式三：指令微调（JSON）
# ============================================================

def _build_instruction(json_path, tokenizer, batch_size, max_length,
                       shuffle, drop_last, num_workers,
                       pad_token_id, ignore_index, device):
    """JSON → Alpaca 格式 + 动态 batch padding DataLoader。"""
    import json

    # 【定制区】替换为你自己的 JSON 加载逻辑
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 【定制区】修改 train/val/test 划分比例
    n = len(data)
    train_data = data[:int(n * 0.85)]
    val_data = data[int(n * 0.85):int(n * 0.95)]
    test_data = data[int(n * 0.95):]

    train_dataset = _InstructionDataset(train_data, tokenizer)
    val_dataset = _InstructionDataset(val_data, tokenizer)
    test_dataset = _InstructionDataset(test_data, tokenizer)

    # 动态 batch padding 的 collate 函数
    def collate_fn(batch):
        return _instruction_collate(batch, pad_token_id, ignore_index,
                                    max_length, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle, drop_last=drop_last,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=True,
                              persistent_workers=num_workers > 0,
                              prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, drop_last=False,
                            num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=True,
                            persistent_workers=num_workers > 0,
                            prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False,
                             num_workers=num_workers, collate_fn=collate_fn,
                             pin_memory=True,
                             persistent_workers=num_workers > 0,
                             prefetch_factor=2)
    return train_loader, val_loader, test_loader


class _InstructionDataset(Dataset):
    """指令数据集 —— 预 tokenize 全部文本。"""

    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            full_text = _format_alpaca(entry)
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, idx):
        return self.encoded_texts[idx]

    def __len__(self):
        return len(self.data)


def _format_alpaca(entry):
    """Alpaca 模板。"""
    instruction = entry.get("instruction", "")
    inp = entry.get("input", "")
    output = entry.get("output", "")
    # 【定制区】换成你自己的 prompt 模板
    text = ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{instruction}")
    if inp:
        text += f"\n\n### Input:\n{inp}"
    text += f"\n\n### Response:\n{output}"
    return text


def _instruction_collate(batch, pad_token_id, ignore_index,
                         max_length, device):
    """按 batch 动态 padding，target 中填充位用 ignore_index 屏蔽。"""
    batch_max = max(len(item) + 1 for item in batch)
    if max_length:
        batch_max = min(batch_max, max_length)

    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max - len(new_item))
        # inputs: 去掉最后一个填充位
        # targets: 右移一位
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        if max_length:
            inputs = inputs[:max_length]
            targets = targets[:max_length]

        # 将首个填充 token 之后的所有目标替换为 ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
