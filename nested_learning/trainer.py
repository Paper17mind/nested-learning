"""Training and evaluation routines for Nested Learning Lite in pure NumPy.

Reference:
- Behrouz et al., "Nested Learning: The Illusion of Deep Learning Architectures", NeurIPS 2025.
- https://github.com/obekt/HOPE-nested-learning
"""

import math
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np

from nested_learning.loss import CrossEntropyLoss
from nested_learning.model import HOPE
from nested_learning.optimizers import NestedOptimizer
from nested_learning.tokenizer import BaseTokenizer


class TextDataset:
    """Sequence dataset for autoregressive language modeling."""

    def __init__(
        self,
        token_ids: Union[List[int], np.ndarray],
        seq_len: int,
        pad_token_id: int = 0,
        stride: Optional[int] = None,
    ):
        self.tokens = np.asarray(token_ids, dtype=np.int64)
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.stride = stride if stride is not None else seq_len

        # Generate slicing start indices
        n_tokens = len(self.tokens)
        if n_tokens <= seq_len:
            self.starts = [0]
        else:
            self.starts = list(range(0, n_tokens - seq_len, self.stride))
    def __len__(self) -> int:
        return len(self.starts)

    @classmethod
    def from_text(
        cls,
        text: str,
        tokenizer: BaseTokenizer,
        seq_len: int = 32,
        stride: Optional[int] = None,
    ) -> "TextDataset":
        """Create a dataset directly from raw text string."""
        token_ids = tokenizer.encode(text)
        return cls(token_ids, seq_len=seq_len, pad_token_id=tokenizer.pad_token_id, stride=stride)

    @classmethod
    def from_file(
        cls,
        filepath: str,
        tokenizer: BaseTokenizer,
        seq_len: int = 32,
        stride: Optional[int] = None,
        encoding: str = "utf-8",
    ) -> "TextDataset":
        """Load text dataset from a plain text file (.txt, .md, etc)."""
        with open(filepath, "r", encoding=encoding, errors="replace") as f:
            text = f.read()
        return cls.from_text(text, tokenizer=tokenizer, seq_len=seq_len, stride=stride)

    @classmethod
    def from_jsonl(
        cls,
        filepath: str,
        tokenizer: BaseTokenizer,
        text_key: str = "text",
        seq_len: int = 32,
        stride: Optional[int] = None,
        encoding: str = "utf-8",
    ) -> "TextDataset":
        """Load dataset from a JSONL file where each line is a JSON object."""
        import json
        texts = []
        with open(filepath, "r", encoding=encoding, errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if text_key in data:
                        texts.append(str(data[text_key]))
        joined_text = "\n\n".join(texts)
        return cls.from_text(joined_text, tokenizer=tokenizer, seq_len=seq_len, stride=stride)

    @classmethod
    def from_qa_pairs(
        cls,
        pairs: List[Dict[str, str]],
        tokenizer: BaseTokenizer,
        seq_len: int = 48,
        stride: Optional[int] = None,
        question_key: str = "question",
        answer_key: str = "answer",
        template: str = "Pertanyaan: {q}\nJawaban: {a}\n\n",
    ) -> "TextDataset":
        """Create dataset from list of question-answer dictionaries."""
        formatted = []
        for item in pairs:
            q = item.get(question_key, "")
            a = item.get(answer_key, "")
            formatted.append(template.format(q=q, a=a))
        joined = "".join(formatted)
        return cls.from_text(joined, tokenizer=tokenizer, seq_len=seq_len, stride=stride)

    def train_val_split(self, val_ratio: float = 0.15) -> Tuple["TextDataset", "TextDataset"]:
        """Split this dataset into train and validation TextDatasets."""
        split_point = int(len(self.tokens) * (1.0 - val_ratio))
        train_tokens = self.tokens[:split_point]
        val_tokens = self.tokens[split_point:]
        train_ds = TextDataset(train_tokens, seq_len=self.seq_len, pad_token_id=self.pad_token_id, stride=self.stride)
        val_ds = TextDataset(val_tokens, seq_len=self.seq_len, pad_token_id=self.pad_token_id, stride=self.stride)
        return train_ds, val_ds
    def get_batches(
        self, batch_size: int, shuffle: bool = True
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield batches of (inputs, targets, mask).

        Inputs: [B, seq_len]
        Targets: [B, seq_len] (shifted by 1)
        Mask: [B, seq_len] (1 = valid, 0 = padding)
        """
        indices = np.arange(len(self.starts))
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            B = len(batch_idx)

            inputs = np.full((B, self.seq_len), self.pad_token_id, dtype=np.int64)
            targets = np.full((B, self.seq_len), self.pad_token_id, dtype=np.int64)
            mask = np.zeros((B, self.seq_len), dtype=np.float32)

            for b, idx in enumerate(batch_idx):
                start = self.starts[idx]
                chunk = self.tokens[start : start + self.seq_len + 1]

                if len(chunk) > 1:
                    inp_slice = chunk[:-1]
                    tgt_slice = chunk[1:]
                    length = len(inp_slice)

                    inputs[b, :length] = inp_slice
                    targets[b, :length] = tgt_slice
                    mask[b, :length] = 1.0

            yield inputs, targets, mask


class Trainer:
    """Trainer for HOPE models with nested multi-frequency optimizer support."""

    def __init__(
        self,
        model: HOPE,
        optimizer: NestedOptimizer,
        loss_fn: Optional[CrossEntropyLoss] = None,
        tokenizer: Optional[BaseTokenizer] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn if loss_fn is not None else CrossEntropyLoss()
        self.tokenizer = tokenizer

        self.history: List[Dict[str, Any]] = []

    def train_step(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        mask: Optional[np.ndarray] = None,
        accumulate_grad: int = 1,
    ) -> float:
        """Run forward, loss, backward, and optionally optimizer step."""
        logits, _ = self.model.forward(inputs, mask=mask)
        loss = self.loss_fn.forward(logits, targets)

        # Analytical backward
        dlogits = self.loss_fn.backward()
        if accumulate_grad > 1:
            dlogits = dlogits / float(accumulate_grad)

        self.model.backward(dlogits)
        return float(loss)

    def evaluate(
        self, dataset: TextDataset, batch_size: int = 4
    ) -> Dict[str, float]:
        """Compute evaluation loss and perplexity on dataset."""
        total_loss = 0.0
        n_batches = 0

        for inputs, targets, mask in dataset.get_batches(batch_size, shuffle=False):
            logits, _ = self.model.forward(inputs, mask=mask)
            loss = self.loss_fn.forward(logits, targets)
            total_loss += loss
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        perplexity = math.exp(min(avg_loss, 20.0))  # guard overflow

        return {
            "loss": float(avg_loss),
            "perplexity": float(perplexity),
            "num_batches": n_batches,
        }

    def train(
        self,
        train_dataset: TextDataset,
        val_dataset: Optional[TextDataset] = None,
        epochs: int = 5,
        batch_size: int = 4,
        accumulate_grad: int = 1,
        max_steps: Optional[int] = None,
        eval_every_steps: int = 50,
        checkpoint_path: Optional[str] = None,
        verbose: bool = True,
        on_step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Full training loop.

        Args:
            train_dataset: Dataset for training.
            val_dataset: Optional dataset for validation.
            epochs: Number of epochs to train.
            batch_size: Micro-batch size.
            accumulate_grad: Number of micro-batches to accumulate before stepping.
            eval_every_steps: Frequency of validation and logging.
            checkpoint_path: Path to save the best model weights (.npz).
            verbose: Whether to print progress.
            on_step_callback: Optional hook called after each optimizer step.

        Returns:
            Dictionary containing training history and best metrics.
        """
        best_val_loss = float("inf")
        step = 0
        running_loss = 0.0
        steps_since_log = 0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            accum_count = 0
            self.optimizer.zero_grad()

            for inputs, targets, mask in train_dataset.get_batches(batch_size, shuffle=True):
                step_loss = self.train_step(
                    inputs, targets, mask=mask, accumulate_grad=accumulate_grad
                )
                running_loss += step_loss
                accum_count += 1

                if accum_count >= accumulate_grad:
                    tier_stepped = self.optimizer.step()
                    self.optimizer.zero_grad()
                    accum_count = 0
                    step += 1
                    steps_since_log += 1

                    # Log / Eval checkpoint
                    if step % eval_every_steps == 0 or step == 1:
                        avg_train_loss = running_loss / max(1, steps_since_log)
                        running_loss = 0.0
                        steps_since_log = 0

                        val_metrics = {}
                        if val_dataset is not None:
                            val_metrics = self.evaluate(val_dataset, batch_size=batch_size)
                            if val_metrics["loss"] < best_val_loss:
                                best_val_loss = val_metrics["loss"]
                                if checkpoint_path:
                                    self.model.save_checkpoint(
                                        checkpoint_path,
                                        extra_meta={
                                            "step": step,
                                            "epoch": epoch,
                                            "val_loss": best_val_loss,
                                        },
                                    )

                        elapsed = time.time() - start_time
                        lr_now = (
                            self.optimizer.scheduler.get_lr(step)
                            if self.optimizer.scheduler
                            else self.optimizer.base_lr
                        )

                        record = {
                            "step": step,
                            "epoch": epoch,
                            "train_loss": avg_train_loss,
                            "lr": lr_now,
                            "tier_stepped": tier_stepped,
                            "elapsed_sec": elapsed,
                            **val_metrics,
                        }
                        self.history.append(record)

                        if verbose:
                            val_str = (
                                f" | val_loss: {val_metrics['loss']:.4f} (ppl: {val_metrics['perplexity']:.2f})"
                                if val_dataset
                                else ""
                            )
                            tiers_str = "".join("1" if s else "0" for s in tier_stepped)
                            print(
                                f"[Step {step:4d} | Ep {epoch}] loss: {avg_train_loss:.4f}{val_str} | "
                                f"lr: {lr_now:.2e} | tiers: [{tiers_str}] | {elapsed:.1f}s"
                            )

                        if on_step_callback:
                            on_step_callback(record)

                    if max_steps is not None and step >= max_steps:
                        break
            if max_steps is not None and step >= max_steps:
                break
        # Flush any trailing accumulated gradients
        if accum_count > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            "total_steps": step,
            "best_val_loss": best_val_loss,
            "history": self.history,
        }
