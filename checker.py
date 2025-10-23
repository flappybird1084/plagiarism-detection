from __future__ import annotations

import re
from itertools import chain
from typing import Iterable, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ollama_embeddings import OllamaEmbedder, OllamaEmbeddingError


class Checked:
    """Compute plagiarism scores for a student submission against provided sources."""

    sentence_splitter = re.compile(r"(?<=[.!?])\s+")
    paragraph_splitter = re.compile(r"\n\s*\n+")
    token_pattern = re.compile(r"\b\w+\b", re.UNICODE)

    def __init__(self, embedder: Optional[OllamaEmbedder] = None) -> None:
        self.embedder = embedder or OllamaEmbedder()

    def analyze(self, student_content: str, source_contents: List[str]) -> dict:
        zero_scores = {
            "token_overlap": 0.0,
            "sentence_similarity_tfidf": 0.0,
            "sentence_similarity_embedding": 0.0,
            "paragraph_similarity_tfidf": 0.0,
            "paragraph_similarity_embedding": 0.0,
        }

        if not source_contents:
            return zero_scores

        aggregate = self._aggregate_scores(student_content, source_contents)
        return {
            "token_overlap": aggregate["token"],
            "sentence_similarity_tfidf": aggregate["sentence_tfidf"],
            "sentence_similarity_embedding": aggregate["sentence_embedding"],
            "paragraph_similarity_tfidf": aggregate["paragraph_tfidf"],
            "paragraph_similarity_embedding": aggregate["paragraph_embedding"],
        }

    def per_source_scores(self, student_content: str, source_contents: List[str]) -> List[dict]:
        per_source = []
        student_sentences = self._split_sentences(student_content)
        student_paragraphs = self._split_paragraphs(student_content)

        for source in source_contents:
            source_sentences = self._split_sentences(source)
            source_paragraphs = self._split_paragraphs(source)
            token_score = self._basic_similarity(student_content, [source])
            sentence_tfidf = self._tfidf_similarity(student_sentences, source_sentences)
            sentence_embedding = self._embedding_similarity(student_sentences, source_sentences)
            paragraph_tfidf = self._tfidf_similarity(student_paragraphs, source_paragraphs)
            paragraph_embedding = self._embedding_similarity(
                student_paragraphs, source_paragraphs
            )
            per_source.append(
                {
                    "token_overlap": token_score,
                    "sentence_similarity_tfidf": sentence_tfidf,
                    "sentence_similarity_embedding": sentence_embedding,
                    "paragraph_similarity_tfidf": paragraph_tfidf,
                    "paragraph_similarity_embedding": paragraph_embedding,
                }
            )
        return per_source

    def _aggregate_scores(self, student_content: str, source_contents: List[str]) -> dict:
        student_sentences = self._split_sentences(student_content)
        source_sentences = list(
            chain.from_iterable(self._split_sentences(text) for text in source_contents)
        )
        student_paragraphs = self._split_paragraphs(student_content)
        paragraph_tfidf_scores: List[float] = []
        paragraph_embedding_scores: List[float] = []
        for source_text in source_contents:
            source_paragraphs = self._split_paragraphs(source_text)
            paragraph_tfidf_scores.append(
                self._tfidf_similarity(student_paragraphs, source_paragraphs)
            )
            paragraph_embedding_scores.append(
                self._embedding_similarity(student_paragraphs, source_paragraphs)
            )

        return {
            "token": self._basic_similarity(student_content, source_contents),
            "sentence_tfidf": self._tfidf_similarity(student_sentences, source_sentences),
            "sentence_embedding": self._embedding_similarity(student_sentences, source_sentences),
            "paragraph_tfidf": self._average_non_zero(paragraph_tfidf_scores),
            "paragraph_embedding": self._average_non_zero(paragraph_embedding_scores),
        }

    def sentence_similarities(
        self, student_content: str, source_content: str
    ) -> List[float]:
        student_sentences = self.split_sentences(student_content)
        if not student_sentences:
            return []

        source_sentences = self.split_sentences(source_content)
        if not source_sentences:
            return [0.0] * len(student_sentences)

        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(student_sentences + source_sentences)
        except ValueError:
            return [0.0] * len(student_sentences)

        split_point = len(student_sentences)
        student_matrix = tfidf_matrix[:split_point]
        source_matrix = tfidf_matrix[split_point:]
        if source_matrix.shape[0] == 0:
            return [0.0] * len(student_sentences)

        similarities = cosine_similarity(student_matrix, source_matrix)
        if similarities.size == 0:
            return [0.0] * len(student_sentences)

        max_per_student = similarities.max(axis=1)
        return max_per_student.tolist()

    def _basic_similarity(self, student_content: str, source_contents: List[str]) -> float:
        student_tokens = set(self._tokenize(student_content))
        source_tokens = set(chain.from_iterable(self._tokenize(text) for text in source_contents))
        if not student_tokens or not source_tokens:
            return 0.0
        intersection = student_tokens & source_tokens
        union = student_tokens | source_tokens
        return len(intersection) / len(union)

    def _tfidf_similarity(self, student_parts: List[str], source_parts: List[str]) -> float:
        student_parts = self._normalize_parts(student_parts)
        source_parts = self._normalize_parts(source_parts)
        if not student_parts or not source_parts:
            return 0.0

        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(student_parts + source_parts)
        except ValueError:
            return 0.0

        split_point = len(student_parts)
        student_matrix = tfidf_matrix[:split_point]
        source_matrix = tfidf_matrix[split_point:]
        if source_matrix.shape[0] == 0:
            return 0.0

        similarities = cosine_similarity(student_matrix, source_matrix)
        if similarities.size == 0:
            return 0.0
        max_per_student = similarities.max(axis=1)
        return float(max_per_student.mean())

    def _embedding_similarity(self, student_parts: List[str], source_parts: List[str]) -> float:
        student_parts = self._normalize_parts(student_parts)
        source_parts = self._normalize_parts(source_parts)
        if not student_parts or not source_parts:
            return 0.0

        try:
            student_vectors = self.embedder.embed_batch(student_parts)
            source_vectors = self.embedder.embed_batch(source_parts)
        except OllamaEmbeddingError:
            return 0.0

        if not student_vectors or not source_vectors:
            return 0.0

        similarities = cosine_similarity(student_vectors, source_vectors)
        if similarities.size == 0:
            return 0.0
        max_per_student = similarities.max(axis=1)
        return float(max_per_student.mean())

    def _split_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        return self.sentence_splitter.split(text.strip())

    def _split_paragraphs(self, text: str) -> List[str]:
        if not text:
            return []
        return self.paragraph_splitter.split(text.strip())

    def _tokenize(self, text: str) -> List[str]:
        return self.token_pattern.findall(text.lower())

    def _normalize_parts(self, parts: Iterable[str]) -> List[str]:
        return [part.strip() for part in parts if part and part.strip()]

    def split_sentences(self, text: str) -> List[str]:
        return self._normalize_parts(self._split_sentences(text))

    def _average_non_zero(self, values: Iterable[float]) -> float:
        filtered = [value for value in values if value]
        if not filtered:
            return 0.0
        return float(sum(filtered) / len(filtered))


checked = Checked()
