"""Grounded answer composition.

The local generator is extractive by design: every sentence in an answer is
copied from a retrieved chunk and paired with a citation. That makes the demo
safe to run without an LLM and makes unsupported-claim behavior easy to test.
In production, the same evidence pack and citation contract can be passed to
an LLM generator hosted in the organization's cloud environment.
"""

from __future__ import annotations

from .models import Answer, Citation, SearchResult
from .text import content_tokens, split_sentences


class GroundedAnswerGenerator:
    """Compose a short answer only from retrieved evidence."""

    def __init__(self, max_sentences: int = 3, max_characters: int = 1_200) -> None:
        self.max_sentences = max_sentences
        self.max_characters = max_characters

    def generate(self, question: str, results: list[SearchResult]) -> Answer:
        if not results:
            return Answer(
                question=question,
                text="I could not find enough evidence in the indexed sources to answer that question.",
                citations=[],
                results=[],
                sufficient_evidence=False,
            )

        question_tokens = content_tokens(question)
        candidates: list[tuple[float, int, int, str, SearchResult]] = []
        seen_sentences: set[str] = set()
        for result_rank, result in enumerate(results):
            evidence_text = _evidence_text(result)
            for sentence_index, sentence in enumerate(split_sentences(evidence_text)):
                normalized = sentence.lower()
                if normalized in seen_sentences or len(sentence) < 35:
                    continue
                seen_sentences.add(normalized)
                overlap = len(question_tokens & content_tokens(sentence))
                # Rank evidence primarily by lexical support, then by the
                # retrieval order. The top retrieved sentence is retained as a
                # fallback when wording differs from the question.
                score = overlap * 10.0 - result_rank - sentence_index * 0.01
                candidates.append((score, result_rank, sentence_index, sentence, result))

        if not candidates:  # pragma: no cover - chunks normally contain sentences
            return Answer(
                question=question,
                text="The retrieved sources did not contain a passage that could be cited for this question.",
                citations=[],
                results=results,
                sufficient_evidence=False,
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        if not any(item[0] > 0 for item in candidates[: self.max_sentences]):
            selected = [min(candidates, key=lambda item: (item[1], item[2]))]
        else:
            best = candidates[0]
            best_chunk_id = best[4].chunk.chunk_id
            best_document_id = best[4].chunk.document_id

            def coherence_key(item: tuple[float, int, int, str, SearchResult]) -> float:
                result = item[4]
                bonus = 0.0
                if result.chunk.chunk_id == best_chunk_id:
                    bonus = 12.0
                elif result.chunk.document_id == best_document_id:
                    bonus = 6.0
                return item[0] + bonus

            remainder = sorted(candidates[1:], key=coherence_key, reverse=True)
            selected = [best, *remainder[: self.max_sentences - 1]]

        citations_by_chunk: dict[str, Citation] = {}
        ordered_citations: list[Citation] = []
        parts: list[str] = []
        total_length = 0
        for _, _, _, sentence, result in selected:
            chunk_id = result.chunk.chunk_id
            if chunk_id not in citations_by_chunk:
                marker = f"[{len(ordered_citations) + 1}]"
                citation = Citation(
                    marker=marker,
                    title=result.chunk.title,
                    source=result.chunk.source,
                    chunk_id=chunk_id,
                    page=result.chunk.metadata.get("page"),
                )
                citations_by_chunk[chunk_id] = citation
                ordered_citations.append(citation)
            part = f"{sentence} {citations_by_chunk[chunk_id].marker}"
            if total_length + len(part) > self.max_characters and parts:
                break
            parts.append(part)
            total_length += len(part)

        return Answer(
            question=question,
            text=" ".join(parts),
            citations=ordered_citations,
            results=results,
            sufficient_evidence=True,
        )


def _evidence_text(result: SearchResult) -> str:
    """Remove titles, headings, and page markers before sentence selection.

    Titles and headings are valuable retrieval metadata, but they are not
    answer sentences. Removing them here keeps citations intact while making
    the extractive output readable.
    """

    import re

    text = re.sub(r"\[Page\s+\d+\]", " ", result.chunk.text)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lstrip("#").strip() == result.chunk.title:
            continue
        if line.startswith("#"):
            continue
        lines.append(line)
    return " ".join(lines)
