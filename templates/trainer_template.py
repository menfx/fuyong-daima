"""
可复用训练/微调模板。

三种模式（通过 mode 参数切换）：
    "pretrain"       — 预训练：全 token 交叉熵 + 每 epoch 生成样本
    "instruction"    — 指令微调：同上，ignore_index 屏蔽填充 token
    "classification" — 分类微调：仅用最后 token + 每 epoch 算准确率

用法：
    from templates.trainer_template import train

    train(model, train_loader, val_loader, optimizer, device,
          mode="classification", num_epochs=5)

定制点：搜索 "【定制区】"，按需求修改对应代码块。
"""

import torch
import torch.nn as nn


# ============================================================
# 核心训练循环
# ============================================================

def train(model, train_loader, val_loader, optimizer, device,
          mode="pretrain", num_epochs=10, eval_freq=50, eval_iter=5,
          tokenizer=None, start_context=None, ignore_index=-100):
    """
    通用训练入口。

    参数：
        model         : nn.Module
        train_loader  : 训练 DataLoader
        val_loader    : 验证 DataLoader
        optimizer     : torch.optim 优化器
        device        : "cpu" / "cuda" / "mps"
        mode          : "pretrain" | "instruction" | "classification"
        num_epochs    : 训练轮数
        eval_freq     : 每 N 步评估一次
        eval_iter     : 评估时用前 N 个 batch
        tokenizer     : (仅 pretrain/instruction 需要)
        start_context : (仅 pretrain/instruction 需要) 生成样本的 prompt
        ignore_index  : (仅 instruction) 填充位置忽略值

    返回：
        train_losses, val_losses  — 评估点记录的损失列表
    """
    train_losses, val_losses = [], []
    global_step = 0
    best_val_loss = float("inf")

    # 【定制区】学习率调度（训练后期收敛困难时取消注释）
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=2)

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            # ---- step 1: 前向 + 损失 ----
            optimizer.zero_grad(set_to_none=True)
            loss = _calc_loss(input_batch, target_batch, model, device,
                              mode, ignore_index)
            # ---- step 2: 反向传播 ----
            loss.backward()
            # ---- step 2.5: 梯度裁剪（深层模型必需，防止梯度爆炸）----
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ---- step 3: 更新参数 ----
            optimizer.step()
            global_step += 1

            # ---- 定期评估 ----
            if global_step % eval_freq == 0:
                train_loss, val_loss = _eval_losses(
                    model, train_loader, val_loader, device,
                    eval_iter, mode, ignore_index)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1:2d} | Step {global_step:6d} | "
                      f"Train loss {train_loss:.3f} | Val loss {val_loss:.3f}")

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "best_model.pth")
                    print(f"  -> New best model saved (val_loss={val_loss:.3f})")

        # ---- epoch 结束回调 ----

        # 【定制区-预训练/指令微调】每 epoch 生成一段文本看效果
        if mode in ("pretrain", "instruction") and tokenizer and start_context:
            _print_sample(model, tokenizer, device, start_context)

        # 【定制区-分类微调】每 epoch 输出准确率
        if mode == "classification":
            train_acc = _calc_accuracy(model, train_loader, device, eval_iter)
            val_acc = _calc_accuracy(model, val_loader, device, eval_iter)
            print(f"Epoch {epoch+1:2d} | Train acc {train_acc:.3f} | "
                  f"Val acc {val_acc:.3f}")

        # 【定制区】学习率调度 step（需配合上面的 scheduler 定义）
        # scheduler.step(val_loss if val_losses else 0)

    # 训练完毕，加载最佳模型
    if best_val_loss < float("inf"):
        model.load_state_dict(torch.load("best_model.pth", weights_only=True))
        print(f"Loaded best model (val_loss={best_val_loss:.3f})")

    return train_losses, val_losses


# ============================================================
# 辅助函数 —— 一般不需要修改
# ============================================================

def _calc_loss(input_batch, target_batch, model, device, mode, ignore_index):
    """计算单 batch 损失。"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)

    if mode == "classification":
        # 分类微调：只取最后一个 token 的 logits
        logits = logits[:, -1, :]
    else:
        # 预训练/指令微调：展平所有 token
        logits = logits.flatten(0, 1)
        target_batch = target_batch.flatten()

    return nn.functional.cross_entropy(logits, target_batch,
                                       ignore_index=ignore_index)


def _eval_losses(model, train_loader, val_loader, device,
                 num_batches, mode, ignore_index):
    """评估训练集和验证集的平均损失。"""
    model.eval()
    with torch.no_grad():
        train_loss = _avg_loss(train_loader, model, device, num_batches,
                               mode, ignore_index)
        val_loss = _avg_loss(val_loader, model, device, num_batches,
                             mode, ignore_index)
    model.train()
    return train_loss, val_loss


def _avg_loss(loader, model, device, num_batches, mode, ignore_index):
    total = 0.0
    for i, (x, y) in enumerate(loader):
        if i >= num_batches:
            break
        total += _calc_loss(x, y, model, device, mode, ignore_index).item()
    return total / num_batches


def _calc_accuracy(model, loader, device, num_batches):
    """计算分类准确率（仅 classification 模式使用）。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
            x, y = x.to(device), y.to(device)
            preds = model(x)[:, -1, :].argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def _print_sample(model, tokenizer, device, start_context, max_len=50):
    """生成一段文本并打印（仅 pretrain/instruction 模式使用）。"""
    model.eval()
    with torch.no_grad():
        ids = tokenizer.encode(start_context, allowed_special={"<|endoftext|>"})
        ids = torch.tensor(ids, device=device).unsqueeze(0)
        for _ in range(max_len):
            # 【定制区】这里的上下文窗口大小按你的模型配置调整
            logits = model(ids[:, -256:])[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        print("  [Sample]", tokenizer.decode(ids[0].tolist())
              .replace("\n", " "))
    model.train()
