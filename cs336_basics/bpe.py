from __future__ import annotations

import multiprocessing as mp
import os
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
import json

import regex
import torch

GPT2_PRETOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
PRETOKEN_RE = regex.compile(GPT2_PRETOKEN_PATTERN)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)

        # Ensure any declared special tokens exist as full tokens in the vocabulary.
        self.special_tokens: list[str] = list(special_tokens or [])
        vocab_values = set(self.vocab.values())
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            if special_bytes not in vocab_values:
                self.vocab[len(self.vocab)] = special_bytes
                vocab_values.add(special_bytes)

        self.bytes_to_id: dict[bytes, int] = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        if self.special_tokens:
            escaped = [regex.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)]
            self.special_pattern = regex.compile("|".join(escaped))
            self.special_to_id = {
                token: self.bytes_to_id[token.encode("utf-8")]
                for token in self.special_tokens
            }
        else:
            self.special_pattern = None
            self.special_to_id: dict[str, int] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]

        if str(vocab_filepath).endswith(".pkl"):
            with open(vocab_filepath, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and "vocab" in payload:
                vocab = payload["vocab"]
                if "merges" in payload:
                    merges = payload["merges"]
                else:
                    with open(merges_filepath, "rb") as f:
                        merges = pickle.load(f)
            else:
                vocab = payload
                with open(merges_filepath, "rb") as f:
                    merges = pickle.load(f)
            return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        with open(merges_filepath, "r", encoding="utf-8") as f:
            raw_merges = [line.strip() for line in f if line.strip()]

        vocab = {}
        for k, v in raw_vocab.items():
            if isinstance(k, str) and k.isdigit():
                vocab[int(k)] = v.encode("utf-8") if isinstance(v, str) else bytes(v)
            else:
                vocab[int(v)] = k.encode("utf-8")

        merges = []
        for line in raw_merges:
            parts = line.split(" ")
            if len(parts) != 2:
                continue
            merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _apply_bpe(self, token_bytes: bytes) -> list[bytes]:
        if not token_bytes:
            return []

        parts = [bytes([b]) for b in token_bytes]
        while len(parts) > 1:
            best_rank = None
            best_pair = None
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            merged_parts: list[bytes] = []
            i = 0
            while i < len(parts):
                if i + 1 < len(parts) and parts[i] == best_pair[0] and parts[i + 1] == best_pair[1]:
                    merged_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged_parts.append(parts[i])
                    i += 1
            parts = merged_parts

        return parts

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in PRETOKEN_RE.finditer(text):
            pretoken = match.group(0).encode("utf-8")
            for token in self._apply_bpe(pretoken):
                ids.append(self.bytes_to_id[token])
        return ids

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        if self.special_pattern is None:
            return self._encode_ordinary(text)

        ids: list[int] = []
        pos = 0
        for match in self.special_pattern.finditer(text):
            if match.start() > pos:
                ids.extend(self._encode_ordinary(text[pos:match.start()]))
            ids.append(self.special_to_id[match.group(0)])
            pos = match.end()
        if pos < len(text):
            ids.extend(self._encode_ordinary(text[pos:]))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        combined = b"".join(self.vocab[token_id] for token_id in ids)
        return combined.decode("utf-8", errors="replace")


def _count_pretokens_in_segment(segment: str) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    for match in PRETOKEN_RE.finditer(segment):
        counts[match.group(0).encode("utf-8")] += 1
    return counts


def _count_pretokens(
    segments: list[str],
    num_workers: int,
) -> Counter[bytes]:
    if num_workers <= 1 or len(segments) <= 1:
        counts: Counter[bytes] = Counter()
        for segment in segments:
            counts.update(_count_pretokens_in_segment(segment))
        return counts

    with mp.Pool(processes=num_workers) as pool:
        partial_counts = pool.map(_count_pretokens_in_segment, segments)

    merged: Counter[bytes] = Counter()
    for part in partial_counts:
        merged.update(part)
    return merged


def _resolve_torch_device(device: str | None) -> str | None:
    requested = (device or "auto").lower()
    if requested in {"none", "off", "cpu"}:
        return None
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else None
    if requested.startswith("cuda"):
        return requested if torch.cuda.is_available() else None
    return None


def _select_best_pair(
    pair_counts: dict[tuple[int, int], int],
    token_bytes: list[bytes],
    torch_device: str | None,
) -> tuple[int, int]:
    pairs = list(pair_counts.keys())
    counts = [pair_counts[p] for p in pairs]

    if torch_device is not None:
        counts_tensor = torch.tensor(counts, device=torch_device, dtype=torch.int64)
        best_count = int(torch.max(counts_tensor).item())
    else:
        best_count = max(counts)

    return max(
        (pair for pair in pairs if pair_counts[pair] == best_count),
        key=lambda p: (token_bytes[p[0]], token_bytes[p[1]]),
    )


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer and return its vocabulary and merge list.

    kwargs:
        workers / num_workers: CPU worker count for pretoken counting.
        device: "auto", "cpu", "cuda", "cuda:0", "off", or "none".
    """
    num_workers = max(1, int(kwargs.get("num_workers", kwargs.get("workers", 1))))
    torch_device = _resolve_torch_device(kwargs.get("device", "cpu"))

    with open(input_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    # Split on special tokens and ignore them during BPE training.
    if special_tokens:
        special_pattern = "|".join(regex.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
        segments = regex.split(special_pattern, corpus)
    else:
        segments = [corpus]

    pretoken_counts = _count_pretokens(segments, num_workers=num_workers)

    # Base vocab always contains all byte values.
    token_bytes: list[bytes] = [bytes([i]) for i in range(256)]
    merges: list[tuple[bytes, bytes]] = []

    target_merges = max(0, vocab_size - 256 - len(special_tokens))
    if target_merges == 0 or not pretoken_counts:
        vocab = {i: tok for i, tok in enumerate(token_bytes)}
        for st in special_tokens:
            vocab[len(vocab)] = st.encode("utf-8")
        return vocab, merges

    # Represent each unique pretoken once with an associated count.
    word_symbols: list[list[int]] = [list(token) for token in pretoken_counts.keys()]
    word_freqs: list[int] = list(pretoken_counts.values())

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)

    def add_word_pairs(word_id: int, symbols: list[int], freq: int) -> None:
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word_id)

    def remove_word_pairs(word_id: int, symbols: list[int], freq: int) -> None:
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            new_count = pair_counts[pair] - freq
            if new_count <= 0:
                pair_counts.pop(pair, None)
            else:
                pair_counts[pair] = new_count

            words_for_pair = pair_to_words.get(pair)
            if words_for_pair is not None:
                words_for_pair.discard(word_id)
                if not words_for_pair:
                    pair_to_words.pop(pair, None)

    for wid, (symbols, freq) in enumerate(zip(word_symbols, word_freqs, strict=True)):
        add_word_pairs(wid, symbols, freq)

    for _ in range(target_merges):
        if not pair_counts:
            break

        best_pair = _select_best_pair(
            pair_counts,
            token_bytes,
            torch_device=torch_device,
        )
        a, b = best_pair
        merged = token_bytes[a] + token_bytes[b]
        new_id = len(token_bytes)
        token_bytes.append(merged)
        merges.append((token_bytes[a], token_bytes[b]))

        affected_words = list(pair_to_words.get(best_pair, ()))
        if not affected_words:
            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)
            continue

        for wid in affected_words:
            old_symbols = word_symbols[wid]
            freq = word_freqs[wid]
            if len(old_symbols) < 2:
                continue

            remove_word_pairs(wid, old_symbols, freq)

            new_symbols: list[int] = []
            i = 0
            n = len(old_symbols)
            while i < n:
                if i + 1 < n and old_symbols[i] == a and old_symbols[i + 1] == b:
                    new_symbols.append(new_id)
                    i += 2
                else:
                    new_symbols.append(old_symbols[i])
                    i += 1

            word_symbols[wid] = new_symbols
            add_word_pairs(wid, new_symbols, freq)

    vocab = {i: tok for i, tok in enumerate(token_bytes)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    return vocab, merges
