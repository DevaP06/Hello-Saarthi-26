"""Generate deterministic alternative search queries for Saarthi's hybrid retriever.

This module is deliberately decoupled from the retrieval pipeline. Call
expand_query() before invoking the existing search_index()/retrieve() function.
"""
from __future__ import annotations
import json
import re


def _clean_queries(items, original: str, limit: int) -> list[str]:
    original_norm = re.sub(r"\s+", " ", original).strip().casefold()
    seen = {original_norm}
    out: list[str] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, str):
            continue
        q = re.sub(r"\s+", " ", item).strip(" \t\r\n-•0123456789.")
        norm = q.casefold()
        if len(q) < 4 or norm in seen:
            continue
        seen.add(norm)
        out.append(q[:240])
        if len(out) >= limit:
            break
    return out


def expand_query(query: str, *, llm=None, max_queries: int = 3) -> list[str]:
    """Return [original, alternatives...]; fail safely to [original].

    `llm` is a LangChain chat model compatible with invoke(messages). If omitted,
    Saarthi's configured fast model is loaded lazily.
    """
    original = (query or "").strip()
    if not original:
        return []
    max_queries = max(0, min(int(max_queries), 5))
    if max_queries == 0:
        return [original]
    try:
        if llm is None:
            from src.utils.llm import chat_llm
            llm = chat_llm(fast=True, temperature=0, max_tokens=300)
        system_text = (
            "You rewrite academic search queries for a university course knowledge base. "
            "Preserve the original intent and technical entities. Return ONLY valid JSON "
            "with shape {\"queries\":[\"...\"]}. Do not answer the question."
        )
        user_text = f"Create at most {max_queries} meaningfully different search queries for:\n{original}"
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
        except ImportError:
            messages = [("system", system_text), ("human", user_text)]
        response = llm.invoke(messages)
        raw = getattr(response, "content", "") or ""
        match = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(match.group(0)) if match else {}
        alternatives = _clean_queries(data.get("queries"), original, max_queries)
        return [original, *alternatives]
    except Exception:
        return [original]
