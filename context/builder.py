import re
from dataclasses import dataclass
from typing import Iterable, List


WORD_RE = re.compile(r"\w+")


@dataclass
class ChatMessage:
    role: str
    content: str


class ContextBuilder:
    """
    Build a prompt under a token budget using role-aware formatting and
    relevance+recency selection over chat history.
    """

    def __init__(self, tokenizer, max_context_tokens: int = 20_480):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens

    def _token_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _words(self, text: str) -> set:
        return set(w.lower() for w in WORD_RE.findall(text))

    def _normalize_history(self, history: Iterable) -> List[ChatMessage]:
        normalized: List[ChatMessage] = []
        for turn in history or []:
            if isinstance(turn, dict):
                role = str(turn.get("role", "")).strip().lower()
                content = str(turn.get("content", "")).strip()
                if role in {"user", "assistant"} and content:
                    normalized.append(ChatMessage(role=role, content=content))
                continue

            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                user_text = str(turn[0]).strip()
                assistant_text = str(turn[1]).strip()
                if user_text:
                    normalized.append(ChatMessage(role="user", content=user_text))
                if assistant_text:
                    normalized.append(ChatMessage(role="assistant", content=assistant_text))
        return normalized

    def _format_message(self, msg: ChatMessage) -> str:
        role = "User" if msg.role == "user" else "Assistant"
        return f"{role}: {msg.content}"

    def _score(self, query_words: set, msg_words: set, index: int, total: int) -> float:
        # Recency prior (more recent turns get larger values).
        recency = (index + 1) / max(total, 1)
        # Lexical relevance proxy (Jaccard) as a cheap semantic signal.
        inter = len(query_words & msg_words)
        union = len(query_words | msg_words) or 1
        relevance = inter / union
        return 0.6 * relevance + 0.4 * recency

    def build(
        self,
        system_message: str,
        user_message: str,
        history=None,
        reserved_output_tokens: int = 512,
    ) -> str:
        system_message = system_message.strip() or "You are a helpful assistant."
        user_message = user_message.strip()

        budget = max(128, self.max_context_tokens - max(1, reserved_output_tokens))
        messages = self._normalize_history(history)

        header = [f"System: {system_message}"]
        fixed_tail = [f"User: {user_message}", "Assistant:"]

        used = self._token_len("\n".join(header + fixed_tail))
        available = max(0, budget - used)

        query_words = self._words(user_message)
        scored = []
        total = len(messages)
        for i, msg in enumerate(messages):
            formatted = self._format_message(msg)
            tok_len = self._token_len(formatted)
            msg_words = self._words(msg.content)
            score = self._score(query_words, msg_words, i, total)
            # Score density helps prioritize compact but useful turns.
            density = score / max(tok_len, 1)
            scored.append((i, formatted, tok_len, density))

        # Pick best messages under budget, then restore original chronology.
        kept_idx = []
        running = 0
        for i, _, tok_len, _ in sorted(scored, key=lambda x: x[3], reverse=True):
            if running + tok_len <= available:
                kept_idx.append(i)
                running += tok_len

        kept_idx_set = set(kept_idx)
        kept = [self._format_message(messages[i]) for i in range(total) if i in kept_idx_set]

        dropped = [messages[i].content for i in range(total) if i not in kept_idx_set]
        summary_block = []
        if dropped:
            compact = []
            for txt in dropped[-3:]:
                sentence = txt.split("\n", 1)[0].strip()
                if len(sentence) > 140:
                    sentence = sentence[:137] + "..."
                if sentence:
                    compact.append(f"- {sentence}")
            if compact:
                summary_block = ["ContextSummary:", *compact]

        prompt_parts = header + summary_block + kept + fixed_tail
        return "\n".join(prompt_parts)
