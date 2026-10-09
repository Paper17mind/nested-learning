"""Zero-dependency tokenizers for Nested Learning Lite in pure Python.

Includes:
- ByteTokenizer: 100% UTF-8 coverage with zero <unk> tokens, fixed small vocab (~260 tokens).
- CharTokenizer: Minimalist character-level tokenizer built from any text corpus.
- SubwordTokenizer: Simple BPE-style tokenizer with merge rules.
"""

import json
from typing import Dict, List, Optional, Set, Tuple, Union


class BaseTokenizer:
    """Base interface for zero-dependency tokenizers."""

    def __init__(self):
        self.pad_token_id: int = 0
        self.eos_token_id: int = 1
        self.bos_token_id: int = 2
        self.unk_token_id: int = 3
        self.vocab_size: int = 4

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        raise NotImplementedError

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        raise NotImplementedError

    def save(self, filepath: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, filepath: str) -> "BaseTokenizer":
        raise NotImplementedError


class ByteTokenizer(BaseTokenizer):
    """Byte-Level Tokenizer for 100% lossless UTF-8 representation with zero dependencies.

    Vocabulary Layout:
    - 0 .. 255: Literal UTF-8 bytes
    - 256: <pad>
    - 257: <eos>
    - 258: <bos>
    - 259: <sep>
    Total vocab size: 260.

    Advantages:
    - Handles Indonesian, English, multilingual text, emojis, and code without any out-of-vocabulary (<unk>) tokens.
    - Extremely lightweight (vocab size = 260), so embeddings and LM heads are ultra fast on CPU.
    """

    def __init__(self):
        super().__init__()
        self.pad_token_id: int = 256
        self.eos_token_id: int = 257
        self.bos_token_id: int = 258
        self.sep_token_id: int = 259
        self.unk_token_id: int = 259
        self.vocab_size: int = 260

        self.special_tokens_set: Set[int] = {
            self.pad_token_id,
            self.eos_token_id,
            self.bos_token_id,
            self.sep_token_id,
        }

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        raw_bytes = text.encode("utf-8")
        tokens = list(raw_bytes)
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        byte_list = bytearray()
        for tok in token_ids:
            if tok in self.special_tokens_set:
                if not skip_special_tokens:
                    if tok == self.pad_token_id:
                        byte_list.extend(b"<pad>")
                    elif tok == self.eos_token_id:
                        byte_list.extend(b"<eos>")
                    elif tok == self.bos_token_id:
                        byte_list.extend(b"<bos>")
                    elif tok == self.sep_token_id:
                        byte_list.extend(b"<sep>")
                continue
            if 0 <= tok < 256:
                byte_list.append(tok)

        return byte_list.decode("utf-8", errors="replace")

    def save(self, filepath: str) -> None:
        data = {
            "type": "byte",
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "sep_token_id": self.sep_token_id,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ByteTokenizer":
        return cls()


class CharTokenizer(BaseTokenizer):
    """Character-Level Tokenizer built dynamically from text corpus."""

    def __init__(self, chars: Optional[List[str]] = None):
        super().__init__()
        self.special_tokens = ["<pad>", "<eos>", "<bos>", "<unk>"]
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3

        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

        for i, s in enumerate(self.special_tokens):
            self.char_to_id[s] = i
            self.id_to_char[i] = s

        if chars:
            for ch in sorted(set(chars)):
                if ch not in self.char_to_id:
                    new_id = len(self.char_to_id)
                    self.char_to_id[ch] = new_id
                    self.id_to_char[new_id] = ch

        self.vocab_size = len(self.char_to_id)

    @classmethod
    def train_from_text(cls, text: str) -> "CharTokenizer":
        unique_chars = sorted(set(text))
        return cls(chars=unique_chars)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens = []
        if add_bos:
            tokens.append(self.bos_token_id)
        for ch in text:
            tokens.append(self.char_to_id.get(ch, self.unk_token_id))
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        chars = []
        for tid in token_ids:
            if skip_special_tokens and tid in (
                self.pad_token_id,
                self.eos_token_id,
                self.bos_token_id,
                self.unk_token_id,
            ):
                continue
            chars.append(self.id_to_char.get(tid, "<unk>"))
        return "".join(chars)

    def save(self, filepath: str) -> None:
        data = {
            "type": "char",
            "char_to_id": self.char_to_id,
            "vocab_size": self.vocab_size,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.char_to_id = {k: int(v) for k, v in data["char_to_id"].items()}
        tok.id_to_char = {int(v): k for k, v in data["char_to_id"].items()}
        tok.vocab_size = len(tok.char_to_id)
        return tok


def get_tokenizer(name: str = "byte", text_corpus: Optional[str] = None) -> BaseTokenizer:
    """Factory helper to obtain a tokenizer instance."""
    if name.lower() == "byte":
        return ByteTokenizer()
    elif name.lower() == "char":
        if text_corpus is not None:
            return CharTokenizer.train_from_text(text_corpus)
        return CharTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type '{name}'. Options: 'byte', 'char'.")
