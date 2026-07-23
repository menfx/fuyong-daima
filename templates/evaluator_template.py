"""
可复用性能评估模板。

支持的评估方式：
    loss        — 交叉熵损失
    accuracy    — 分类准确率（取最后一个 token）
    perplexity  — 困惑度 = exp(loss)
    generate    — 给定 prompt 生成文本
    judge       — LLM-as-a-Judge 自动评分（需要 ollama 运行中）

用法：
    from templates.evaluator_template import evaluate

    results = evaluate(model, data_loader, device,
                       metrics=["loss", "accuracy"])

定制点：搜索 "【定制区】"，按需求修改对应代码块。
"""

import torch
import torch.nn as nn
import json


# ============================================================
# 核心评估入口
# ============================================================

def evaluate(model, data_loader, device, tokenizer=None, metrics=None,
             prompts=None, judge_model="llama3",
             ollama_url="http://localhost:11434/api/chat",
             task="pretrain"):
    """
    统一评估入口。

    参数：
        model        : nn.Module（应为 eval() 状态）
        data_loader  : DataLoader
        device       : "cpu" / "cuda"
        tokenizer    : (generate / judge 需要)
        metrics      : ["loss","accuracy","perplexity","generate","judge"] 的子集
        prompts      : (generate) 要生成文本的 prompt 列表
        judge_model  : (judge) ollama 中拉取的模型名
        ollama_url   : (judge) ollama API 地址
        task         : "pretrain" (全 token CE) | "classification" (仅最后 token) 

    返回：
        dict，如 {"loss": 0.35, "accuracy": 0.97, "perplexity": 1.42}
    """
    if metrics is None:
        metrics = ["loss"]

    results = {}
    model.eval()

    with torch.no_grad():
        if "loss" in metrics:
            results["loss"] = _calc_avg_loss(model, data_loader, device, task)

        if "accuracy" in metrics:
            results["accuracy"] = _calc_accuracy(model, data_loader, device)

        if "perplexity" in metrics:
            loss = results.get("loss", _calc_avg_loss(model, data_loader, device, task))
            results["perplexity"] = torch.exp(torch.tensor(loss)).item()

        if "generate" in metrics and tokenizer and prompts:
            results["generations"] = _generate_texts(model, tokenizer, prompts, device)

        if "judge" in metrics and "generations" in results:
            results["judge_scores"] = _judge_generations(
                results["generations"], judge_model, ollama_url)

    return results


# ============================================================
# 各指标计算函数 —— 一般不需要修改
# ============================================================

def _calc_avg_loss(model, loader, device, task="pretrain"):
    """计算平均交叉熵损失。
    
    task="pretrain"       → 全 token 展平后 CE（预训练/指令微调）
    task="classification" → 仅最后一个 token 做 CE（分类微调）
    """
    total_loss, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if task == "classification":
            logits = model(x)[:, -1, :]
        else:
            logits = model(x).flatten(0, 1)
            y = y.flatten()
        loss = nn.functional.cross_entropy(logits, y)
        total_loss += loss.item()
        count += 1
    return total_loss / count


def _calc_accuracy(model, loader, device):
    """计算分类准确率（取最后一个 token 做分类）。"""
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)[:, -1, :].argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def _generate_texts(model, tokenizer, prompts, device, max_len=60):
    """对给定 prompts 生成文本。"""
    generated = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        ids = torch.tensor(ids, device=device).unsqueeze(0)
        for _ in range(max_len):
            # 【定制区】调整上下文窗口大小和采样策略
            logits = model(ids[:, -256:])[:, -1, :]
            # 贪心解码（需要随机性时改为 temperature + top-k）
            next_id = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        generated.append({
            "prompt": prompt,
            "output": tokenizer.decode(ids[0].tolist()),
        })
    return generated


def _judge_generations(generations, judge_model, ollama_url):
    """用 Llama 3 对生成结果打分（需要 ollama 运行中）。

    【定制区】修改评分 prompt 或评分标准。
    """
    import requests

    scores = []
    for item in generations:
        judge_prompt = (
            f"Score the following response on a scale from 1 to 100 "
            f"based on accuracy and fluency.\n\n"
            f"Prompt: {item['prompt']}\n"
            f"Response: {item['output']}\n\n"
            f"Reply with ONLY the integer score."
        )
        try:
            r = requests.post(
                ollama_url,
                json={
                    "model": judge_model,
                    "messages": [{"role": "user", "content": judge_prompt}],
                    "stream": False,
                },
                timeout=120,
            )
            r.raise_for_status()
            score = r.json()["message"]["content"].strip()
            scores.append(int(score))
        except Exception as e:
            scores.append(None)
            print(f"  [Judge error] {e}")

    return scores


# ============================================================
# 便捷函数
# ============================================================

def evaluate_multiple_loaders(model, loaders, device, tokenizer=None,
                              metrics=None, task="pretrain"):
    """
    对多份数据（训练/验证/测试）同时评估。

    用法：
        results = evaluate_multiple_loaders(
            model,
            {"train": train_loader, "val": val_loader, "test": test_loader},
            device,
            metrics=["loss", "accuracy"],
            task="classification",
        )
    """
    return {
        name: evaluate(model, loader, device, tokenizer, metrics, task=task)
        for name, loader in loaders.items()
    }
