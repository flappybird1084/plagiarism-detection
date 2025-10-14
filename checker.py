from __future__ import annotations

import re
from itertools import chain
from typing import Iterable, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Checked:
    """Compute plagiarism scores for a student submission against provided sources."""

    sentence_splitter = re.compile(r"(?<=[.!?])\s+")
    paragraph_splitter = re.compile(r"\n\s*\n+")
    token_pattern = re.compile(r"\b\w+\b", re.UNICODE)

    def analyze(self, student_content: str, source_contents: List[str]) -> dict:
        if not source_contents:
            return {
                "plagiarism_1": 0.0,
                "plagiarism_2": 0.0,
                "plagiarism_3": 0.0,
            }

        aggregate = self._aggregate_scores(student_content, source_contents)
        return {
            "plagiarism_1": aggregate["token"],
            "plagiarism_2": aggregate["sentence"],
            "plagiarism_3": aggregate["paragraph"],
        }

    def per_source_scores(self, student_content: str, source_contents: List[str]) -> List[dict]:
        per_source = []
        student_sentences = self._split_sentences(student_content)
        student_paragraphs = self._split_paragraphs(student_content)

        for source in source_contents:
            token_score = self._basic_similarity(student_content, [source])
            sentence_score = self._tfidf_similarity(
                student_sentences, self._split_sentences(source)
            )
            paragraph_score = self._tfidf_similarity(
                student_paragraphs, self._split_paragraphs(source)
            )
            per_source.append(
                {
                    "token": token_score,
                    "sentence": sentence_score,
                    "paragraph": paragraph_score,
                }
            )
        return per_source

    def _aggregate_scores(self, student_content: str, source_contents: List[str]) -> dict:
        student_sentences = self._split_sentences(student_content)
        source_sentences = list(
            chain.from_iterable(self._split_sentences(text) for text in source_contents)
        )
        student_paragraphs = self._split_paragraphs(student_content)
        source_paragraphs = list(
            chain.from_iterable(self._split_paragraphs(text) for text in source_contents)
        )

        return {
            "token": self._basic_similarity(student_content, source_contents),
            "sentence": self._tfidf_similarity(student_sentences, source_sentences),
            "paragraph": self._tfidf_similarity(student_paragraphs, source_paragraphs),
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


checked = Checked()
