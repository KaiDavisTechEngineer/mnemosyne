"""MNEMOSYNE tokenizer.

The tokenizer is byte-level with reserved structured tokens for the
agent communication protocol. We deliberately don't ship a BPE merge
table — at MNEMOSYNE's scale (sub-million-parameter agents trained
from scratch) byte-level encoding works fine and keeps the
implementation transparent.

Reserved special tokens
-----------------------
* ``<pad>`` / ``<bos>`` / ``<eos>`` — standard.
* ``<role:proposer>`` / ``<role:critic>`` / ``<role:verifier>`` /
  ``<role:synthesizer>`` / ``<role:metacognitor>`` — identify the
  speaking agent in multi-agent dialogues.
* ``<msg>`` / ``</msg>`` — message envelope.
* ``<ref:N>`` where N is 0-15 — reference to memory slot N (the
  agent can attend over its hierarchical memory).
* ``<intervene>`` — signals that the next emission is a causal
  intervention plan, not a normal utterance.
* ``<introspect>`` — agent is about to report on its own features.
* ``<feature:K>`` where K is 0-255 — reference to one of 256 SAE
  features. The reserved range gives the model a vocabulary for
  *naming its own internal features*, which is the linguistic
  ingredient of causal self-modeling.
"""
from __future__ import annotations

from dataclasses import dataclass


# Reserved special tokens, in stable order. Indices 0..N-1 are these
# specials; the rest of the vocab is byte values shifted by len(SPECIALS).
SPECIALS: tuple[str, ...] = (
    "<pad>", "<bos>", "<eos>",
    "<role:proposer>", "<role:critic>", "<role:verifier>",
    "<role:synthesizer>", "<role:metacognitor>",
    "<msg>", "</msg>",
    "<intervene>", "<introspect>",
    "<ref:0>", "<ref:1>", "<ref:2>", "<ref:3>",
    "<ref:4>", "<ref:5>", "<ref:6>", "<ref:7>",
    "<ref:8>", "<ref:9>", "<ref:10>", "<ref:11>",
    "<ref:12>", "<ref:13>", "<ref:14>", "<ref:15>",
)
# Reserve indices for feature references too: <feature:0> .. <feature:255>
# Will be appended after SPECIALS in the vocab.

N_FEATURES = 256


@dataclass
class Tokenizer:
    """Byte-level tokenizer with structured agent-protocol tokens."""
    specials: tuple[str, ...]
    feature_tokens: tuple[str, ...]

    @property
    def n_specials(self) -> int:
        return len(self.specials) + len(self.feature_tokens)

    @property
    def vocab_size(self) -> int:
        return self.n_specials + 256  # specials + 256 byte values

    @classmethod
    def build(cls) -> "Tokenizer":
        feats = tuple(f"<feature:{i}>" for i in range(N_FEATURES))
        return cls(specials=SPECIALS, feature_tokens=feats)

    # ──────────────────────────────────────────────────────────────────
    # Encoding
    # ──────────────────────────────────────────────────────────────────
    def encode(self, text: str) -> list[int]:
        """Encode text. Special tokens (anything matching ``<...>`` exactly
        from our reserved set) are matched greedily; the rest is encoded
        byte-by-byte as UTF-8."""
        out: list[int] = []
        i = 0
        all_specials = self.specials + self.feature_tokens
        all_special_ids = {s: idx for idx, s in enumerate(all_specials)}
        while i < len(text):
            matched = False
            if text[i] == "<":
                # Try matching a special.
                end = text.find(">", i)
                if end != -1:
                    candidate = text[i:end + 1]
                    if candidate in all_special_ids:
                        out.append(all_special_ids[candidate])
                        i = end + 1
                        matched = True
            if matched:
                continue
            # Byte-level fallback.
            ch = text[i]
            for b in ch.encode("utf-8"):
                out.append(self.n_specials + b)
            i += 1
        return out

    def decode(self, ids: list[int]) -> str:
        """Decode back to a string. Specials are rendered as their
        bracket form; bytes are reassembled as UTF-8 (with errors='replace')."""
        all_specials = self.specials + self.feature_tokens
        out_parts: list[str] = []
        byte_buffer: list[int] = []

        def flush_bytes() -> None:
            if byte_buffer:
                out_parts.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for i in ids:
            if 0 <= i < len(all_specials):
                flush_bytes()
                out_parts.append(all_specials[i])
            elif self.n_specials <= i < self.n_specials + 256:
                byte_buffer.append(i - self.n_specials)
        flush_bytes()
        return "".join(out_parts)

    # ──────────────────────────────────────────────────────────────────
    # Useful accessors
    # ──────────────────────────────────────────────────────────────────
    def special_id(self, name: str) -> int:
        all_specials = self.specials + self.feature_tokens
        return all_specials.index(name)

    def role_token(self, role: str) -> str:
        return f"<role:{role}>"

    def feature_token(self, idx: int) -> str:
        return f"<feature:{idx}>"

    def is_role_token(self, idx: int) -> bool:
        all_specials = self.specials + self.feature_tokens
        if not 0 <= idx < len(all_specials):
            return False
        return all_specials[idx].startswith("<role:")
