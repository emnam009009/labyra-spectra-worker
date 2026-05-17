"""Integration test for src/papers/metadata.py (R177-1d) book detection.

Tests Gemini-based extract_metadata() correctly classifies documentType
from page-1 OCR signals. Real Gemini API call (skip if no key).

@phase R177-1f
"""
from __future__ import annotations

import os
import pytest

from src.papers.metadata import extract_metadata

_HAS_GEMINI_KEY = bool(os.environ.get("GEMINI_API_KEY"))
_REQUIRES_GEMINI = pytest.mark.skipif(
    not _HAS_GEMINI_KEY, reason="GEMINI_API_KEY not set"
)


ARTICLE_SAMPLE = """# Tungsten Trioxide WO3 Nanostructures: Synthesis and Photocatalytic Activity

Nguyen Van A, Tran Thi B, Le Quoc C*

Department of Materials Science, Ho Chi Minh City University of Technology (HCMUT), Vietnam
*Corresponding author: lqc@hcmut.edu.vn

Journal of Materials Chemistry A, Vol. 12, Issue 4, pp. 1234-1245
Published: 15 March 2024
DOI: https://doi.org/10.1021/acsami.4c01234

## Abstract

We report the hydrothermal synthesis of hexagonal WO3 nanorods..."""

BOOK_SAMPLE = """# Infrared and Raman Spectroscopy
## Methods and Applications

Bernhard Schrader (Editor)

Wiley-VCH Verlagsgesellschaft mbH

ISBN 3-527-26446-9

First Edition, 1995

## Table of Contents

Chapter 1: Introduction
Chapter 2: Theory of Vibrational Spectroscopy
Chapter 3: Instrumentation
Chapter 4: Sample Preparation

## Preface

This book provides a comprehensive introduction to infrared and Raman
spectroscopy for chemistry, physics and materials science students..."""


@_REQUIRES_GEMINI
class TestArticleDetection:
    def test_article_documenttype(self) -> None:
        result = extract_metadata(ARTICLE_SAMPLE)
        assert result.document_type == "article"
        assert result.doi == "10.1021/acsami.4c01234"
        assert result.year == 2024
        assert "Tungsten" in result.title
        assert len(result.authors) >= 1
        # Article should NOT have ISBN/publisher
        assert result.isbn == ""
        assert result.publisher == ""


@_REQUIRES_GEMINI
class TestBookDetection:
    def test_book_documenttype(self) -> None:
        result = extract_metadata(BOOK_SAMPLE)
        assert result.document_type == "book"
        assert result.year == 1995
        assert "Infrared" in result.title
        assert "Raman" in result.title
        assert "Schrader" in " ".join(result.authors)
        # Book SHOULD have ISBN + publisher
        assert "527" in result.isbn  # partial match handles hyphens
        assert "Wiley" in result.publisher or "VCH" in result.publisher
        # Book should NOT have DOI
        assert result.doi == ""


class TestDefaults:
    """No API key needed — defaults path."""

    def test_empty_input(self) -> None:
        result = extract_metadata("")
        assert result.title == "Untitled"
        assert result.authors == []
        assert result.year == 0
        assert result.doi == ""
        assert result.document_type == "unknown"
        assert result.isbn == ""
        assert result.publisher == ""

    def test_short_input(self) -> None:
        # < MIN_INPUT_CHARS (50)
        result = extract_metadata("Hello world")
        assert result.title == "Untitled"
        assert result.document_type == "unknown"
