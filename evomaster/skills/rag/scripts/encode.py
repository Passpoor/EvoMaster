#!/usr/bin/env python3
"""Text Encoder - 文本编码工具

提供独立的文本编码功能，将文本转换为向量。
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class TextEncoder:
    """文本编码器"""

    def __init__(
        self,
        model_name: str = "evomaster/skills/rag/local_models/all-mpnet-base-v2",
        device: str = "cpu"
    ):
        """初始化编码器

        Args:
            model_name: Transformer 模型名称
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.model_name = model_name
        self.device = device

        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info(f"Initialized encoder with model: {model_name} on {device}")

    def encode(
        self,
        text: str,
        max_length: int = 512,
        normalize: bool = False
    ) -> np.ndarray:
        """编码文本

        Args:
            text: 输入文本
            max_length: 最大长度
            normalize: 是否归一化向量

        Returns:
            编码后的向量
        """
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            h = outputs.last_hidden_state
            attn = inputs["attention_mask"].unsqueeze(-1)
            # Mean pooling with attention weights
            emb = (h * attn).sum(dim=1) / attn.sum(dim=1)

        emb = emb.cpu().numpy()

        # 归一化（可选）
        if normalize:
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / (norm + 1e-8)

        return emb[0]  # 返回第一个（也是唯一的）向量

    def encode_batch(
        self,
        texts: list[str],
        max_length: int = 512,
        normalize: bool = False,
        batch_size: int = 32
    ) -> np.ndarray:
        """批量编码文本

        Args:
            texts: 文本列表
            max_length: 最大长度
            normalize: 是否归一化向量
            batch_size: 批处理大小

        Returns:
            编码后的向量数组 (n_texts, embedding_dim)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**batch_inputs)
                h = outputs.last_hidden_state
                attn = batch_inputs["attention_mask"].unsqueeze(-1)
                # Mean pooling with attention weights
                emb = (h * attn).sum(dim=1) / attn.sum(dim=1)

            emb = emb.cpu().numpy()

            # 归一化（可选）
            if normalize:
                norm = np.linalg.norm(emb, axis=1, keepdims=True)
                emb = emb / (norm + 1e-8)

            all_embeddings.append(emb)

        return np.vstack(all_embeddings)


def main():
    """命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="Text Encoder CLI")
    parser.add_argument("--model", 
                       default="evomaster/skills/rag/local_models/all-mpnet-base-v2",
                       help="Transformer model path or HuggingFace model name (default: local model)")
    parser.add_argument("--text", help="Text to encode")
    parser.add_argument("--file", help="File containing text (one per line)")
    parser.add_argument("--output", help="Output file for embeddings (.npy)")
    parser.add_argument("--max_length", type=int, default=512, help="Max length")
    parser.add_argument("--normalize", action="store_true", help="Normalize vectors")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    # 初始化编码器
    encoder = TextEncoder(model_name=args.model)

    # 读取文本
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # 从 stdin 读取
        texts = [line.strip() for line in sys.stdin if line.strip()]

    if not texts:
        print("Error: No text provided", file=sys.stderr)
        sys.exit(1)

    # 编码
    if len(texts) == 1:
        embedding = encoder.encode(texts[0], max_length=args.max_length, normalize=args.normalize)
    else:
        embedding = encoder.encode_batch(
            texts,
            max_length=args.max_length,
            normalize=args.normalize,
            batch_size=args.batch_size
        )

    # 输出
    if args.output:
        np.save(args.output, embedding)
        print(f"Saved embeddings to {args.output}")
    else:
        # 输出到 stdout（以可读格式）
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding:\n{embedding}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
