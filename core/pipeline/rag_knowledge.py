"""
rag_knowledge.py — Generic semantic RAG retrieval.
====================================================
KB path is provided via configure_rag() from the use-case config.
No hardcoded paths. Supports extra KB dirs for supplementary JSON files.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import numpy as np
import torch

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[RAG] faiss not available. Using keyword fallback.")

STOPWORDS = {
    "a", "an", "the", "is", "are", "what", "why", "how", "when", "where",
    "i", "me", "my", "you", "your", "do", "does", "did", "can", "will",
    "to", "for", "of", "on", "in", "and", "or", "hai", "kya", "ka", "ki",
}

# ── Module-level config ───────────────────────────────────────────────────────
_KB_PATH: str = ""
_EXTRA_KB_DIRS: list[str] = []
_RAG_THRESHOLD: float = 0.55
_retriever: Optional["RAGRetriever"] = None


def configure_rag(
    kb_path: str,
    *,
    extra_kb_dirs: list[str] | None = None,
    threshold: float = 0.55,
) -> None:
    """Call at startup with use-case config."""
    global _KB_PATH, _EXTRA_KB_DIRS, _RAG_THRESHOLD, _retriever
    _KB_PATH = kb_path
    _EXTRA_KB_DIRS = extra_kb_dirs or []
    _RAG_THRESHOLD = threshold
    _retriever = None   # force re-init


class RAGRetriever:
    """FAQ + structured topic retrieval with confidence scores."""

    def __init__(self, kb_path: str, threshold: float = 0.55, extra_kb_dirs: list[str] | None = None):
        self.threshold = threshold
        self.kb_path   = kb_path
        self.topics:    dict = {}
        self.faqs:      list = []
        self.questions: list[str] = []
        self.answers:   list[str] = []
        self.faq_topics: list[str] = []
        self.index      = None
        self.model      = None
        self.tokenizer  = None
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"

        self._load_kb(kb_path)

        # Load supplementary KB dirs
        for extra_dir in (extra_kb_dirs or []):
            self._load_extra_kb_dir(extra_dir)

        if self._can_use_semantic_rag():
            try:
                self._build_index()
            except Exception as e:
                self.model = None
                self.index = None
                print(f"[RAG] FAISS setup failed: {e}. Falling back to keyword retrieval.")

    # ── KB loading ────────────────────────────────────────────────────────────

    def _load_kb(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.topics    = data.get("topics", {})
        self.faqs      = data.get("faqs", [])
        self.questions = [faq["question"] for faq in self.faqs]
        self.answers   = [faq["answer"]   for faq in self.faqs]
        self.faq_topics = [faq.get("topic", "") for faq in self.faqs]
        print(f"[RAG] Loaded {len(self.faqs)} FAQs from {path}")

    def _load_extra_kb_dir(self, kb_dir: str) -> None:
        """Load supplementary JSON files from an extra KB directory."""
        if not os.path.isdir(kb_dir):
            return
        added = 0
        for fn in os.listdir(kb_dir):
            if not fn.lower().endswith(".json"):
                continue
            full = os.path.join(kb_dir, fn)
            try:
                with open(full, "r", encoding="utf-8") as f:
                    mdata = json.load(f)
            except Exception:
                continue

            topic_key = mdata.get("topic") or os.path.splitext(fn)[0].lower()
            desc = mdata.get("description") or mdata.get("summary")
            if desc:
                if topic_key in self.topics and isinstance(self.topics[topic_key], dict):
                    self.topics[topic_key].setdefault("summary", desc)
                else:
                    self.topics[topic_key] = {"summary": desc}

            faqs = mdata.get("faqs") or mdata.get("faqs_list") or []
            for faq in faqs:
                answer   = faq.get("answer") or faq.get("response") or ""
                examples = faq.get("examples") or faq.get("questions") or []
                question = ""
                if isinstance(examples, list) and examples:
                    question = examples[0]
                else:
                    question = faq.get("intent") or faq.get("question") or ""
                if not question or not answer:
                    continue
                self.faqs.append({"question": question, "answer": answer, "topic": topic_key})
                self.questions.append(question)
                self.answers.append(answer)
                self.faq_topics.append(topic_key)
                added += 1

        if added:
            print(f"[RAG] Merged {added} extra FAQs from {kb_dir}")

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _build_index(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        print("[RAG] Building FAISS semantic index...")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        embeddings = self._encode_texts(self.questions)
        dim        = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.array(embeddings, dtype="float32"))
        print(f"[RAG] FAISS index built with {self.index.ntotal} vectors.")

    def _can_use_semantic_rag(self) -> bool:
        if not FAISS_AVAILABLE or not self.questions:
            return False
        if os.getenv("RAG_DISABLE_SEMANTIC", "").strip().lower() in {"1", "true", "yes"}:
            return False
        try:
            from transformers import AutoModel, AutoTokenizer  # noqa
        except Exception as e:
            print(f"[RAG] transformers not available: {e}")
            return False
        return True

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 1) -> str | None:
        answer, score, _ = self.retrieve_with_confidence(query, top_k=top_k)
        return answer if score >= self.threshold else None

    def retrieve_with_confidence(self, query: str, top_k: int = 3) -> tuple[str | None, float, str | None]:
        if not query or not query.strip():
            return None, 0.0, None

        topic_hint = self._keyword_topic_hint(query)
        if topic_hint:
            answer = self.generate_topic_answer(topic_hint)
            if answer:
                return answer, 0.85, topic_hint

        if self.model and self.index:
            return self._faiss_retrieve_with_confidence(query, top_k)

        answer, score = self._keyword_retrieve_with_score(query)
        return answer, score, topic_hint

    def _faiss_retrieve_with_confidence(self, query: str, top_k: int) -> tuple[str | None, float, str | None]:
        try:
            query_vec = self._encode_texts([query])
            scores, indices = self.index.search(
                np.array(query_vec, dtype="float32"), min(top_k, len(self.questions))
            )
        except Exception as e:
            print(f"[RAG] FAISS retrieval failed: {e}")
            answer, score = self._keyword_retrieve_with_score(query)
            return answer, score, None

        best_score = float(scores[0][0])
        best_idx   = int(indices[0][0])
        print(f"[RAG] Best match: '{self.questions[best_idx]}' (score={best_score:.3f})")

        if best_score >= self.threshold:
            return self.answers[best_idx], best_score, self.faq_topics[best_idx] or None
        return None, best_score, self.faq_topics[best_idx] or None

    def _keyword_retrieve_with_score(self, query: str) -> tuple[str | None, float]:
        normalized_query = self._normalize_text(query)
        query_tokens     = self._tokenize(query)
        best_score       = 0.0
        best_answer: str | None = None

        for q, a in zip(self.questions, self.answers):
            if self._normalize_text(q) == normalized_query:
                return a, 1.0
            question_tokens = self._tokenize(q)
            if not query_tokens or not question_tokens:
                continue
            intersection = len(query_tokens & question_tokens)
            union        = len(query_tokens | question_tokens)
            score        = intersection / union if union else 0.0
            if score > best_score:
                best_score  = score
                best_answer = a

        return (best_answer, best_score) if best_score >= 0.65 else (None, best_score)

    def _keyword_topic_hint(self, query: str) -> str | None:
        """
        Generic topic hints — common across most voice agents.
        Use-case-specific hints should be added to keywords.json.
        """
        t = query.lower()
        hints = {
            "placements": ["placement", "package", "recruit", "salary", "job", "company"],
            "fees":       ["fee", "fees", "cost", "scholarship", "payment", "kitna"],
            "location":   ["location", "address", "where", "campus", "kahan"],
            "hostel":     ["hostel", "accommodation", "stay"],
        }
        for topic, keys in hints.items():
            if any(k in t for k in keys):
                return topic
        return None

    def generate_topic_answer(self, topic_key: str, context: dict | None = None) -> str:
        block = self.topics.get(topic_key)
        if not block:
            return ""
        if isinstance(block, dict) and block.get("summary"):
            return block["summary"]
        if isinstance(block, str):
            return block
        return ""

    def get_step_script(self, step: str) -> str | None:
        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scripts = data.get("step_scripts_outbound") or data.get("step_scripts", {})
        return scripts.get(step)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
        mask       = encoded["attention_mask"]
        token_embs = output.last_hidden_state
        mask_exp   = mask.unsqueeze(-1).expand(token_embs.size()).float()
        pooled     = (token_embs * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.cpu().numpy().astype("float32")

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        return " ".join(normalized.split())

    def _tokenize(self, text: str) -> set[str]:
        return {t for t in self._normalize_text(text).split() if t not in STOPWORDS}


# ── Singleton ─────────────────────────────────────────────────────────────────

def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        if not _KB_PATH:
            raise RuntimeError("[RAG] configure_rag() was not called before get_retriever()")
        _retriever = RAGRetriever(
            kb_path=_KB_PATH,
            threshold=_RAG_THRESHOLD,
            extra_kb_dirs=_EXTRA_KB_DIRS,
        )
    return _retriever
