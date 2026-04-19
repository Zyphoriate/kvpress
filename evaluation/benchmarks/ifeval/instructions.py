# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Rule-based instruction checkers for IFEval.
# Based on google-research/instruction_following_eval
# (commit aa633e5105c702b47a4dd836d9b6eca39984a0fe).

import json
import re
import string
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMPARISON_RELATIONS = {
    "at least": lambda a, b: a >= b,
    "at most": lambda a, b: a <= b,
    "exactly": lambda a, b: a == b,
    "less than": lambda a, b: a < b,
    "more than": lambda a, b: a > b,
}


def _compare(value: int, relation: str, target: int) -> bool:
    fn = _COMPARISON_RELATIONS.get(relation)
    if fn is None:
        return False
    return fn(value, target)


def _count_words(text: str) -> int:
    return len(text.split())


def _count_sentences(text: str) -> int:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return len([s for s in sentences if s.strip()])


def _get_paragraphs(text: str) -> list:
    paras = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paras if p.strip()]


# ---------------------------------------------------------------------------
# Keyword instructions
# ---------------------------------------------------------------------------


def check_keywords_existence(response: str, keywords: list, **kwargs) -> bool:
    """All given keywords must appear in the response (case-insensitive)."""
    response_lower = response.lower()
    return all(kw.lower() in response_lower for kw in keywords)


def check_keywords_frequency(response: str, keyword: str, frequency: int, relation: str, **kwargs) -> bool:
    """Keyword must appear with the required frequency."""
    count = response.lower().count(keyword.lower())
    return _compare(count, relation, frequency)


def check_keywords_forbidden_words(response: str, forbidden_words: list, **kwargs) -> bool:
    """None of the forbidden words may appear in the response (case-insensitive)."""
    response_lower = response.lower()
    return not any(fw.lower() in response_lower for fw in forbidden_words)


def check_keywords_letter_frequency(
    response: str, letter: str, let_frequency: int, let_relation: str, **kwargs
) -> bool:
    """A specific letter must appear with the required frequency."""
    count = response.lower().count(letter.lower())
    return _compare(count, let_relation, let_frequency)


# ---------------------------------------------------------------------------
# Language instruction
# ---------------------------------------------------------------------------


def check_language_response_language(response: str, language: str, **kwargs) -> bool:
    """Response must be in the specified language (ISO 639-1 code).

    Falls back to True when *langdetect* is not installed to avoid hard
    failures in environments that don't have the package.
    """
    try:
        from langdetect import detect  # type: ignore[import]

        detected = detect(response)
        return detected == language
    except ImportError:
        # langdetect not installed – skip check
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Length constraints
# ---------------------------------------------------------------------------


def check_length_number_sentences(response: str, num_sentences: int, relation: str, **kwargs) -> bool:
    count = _count_sentences(response)
    return _compare(count, relation, num_sentences)


def check_length_number_paragraphs(response: str, num_paragraphs: int, **kwargs) -> bool:
    count = len(_get_paragraphs(response))
    return count >= num_paragraphs


def check_length_number_words(response: str, num_words: int, relation: str, **kwargs) -> bool:
    count = _count_words(response)
    return _compare(count, relation, num_words)


def check_length_nth_paragraph_first_word(
    response: str, num_paragraphs: int, nth_paragraph: int, first_word: str, **kwargs
) -> bool:
    """The nth paragraph must start with the given word and total paragraphs
    must equal *num_paragraphs*."""
    paragraphs = _get_paragraphs(response)
    if len(paragraphs) < num_paragraphs:
        return False
    if nth_paragraph < 1 or nth_paragraph > len(paragraphs):
        return False
    first = paragraphs[nth_paragraph - 1].split()[0] if paragraphs[nth_paragraph - 1].split() else ""
    return first.lower().strip(string.punctuation) == first_word.lower().strip(string.punctuation)


# ---------------------------------------------------------------------------
# Detectable content
# ---------------------------------------------------------------------------


def check_detectable_content_number_placeholders(response: str, num_placeholders: int, **kwargs) -> bool:
    """Response must contain at least *num_placeholders* placeholder tokens
    of the form ``[something]``."""
    placeholders = re.findall(r"\[.+?\]", response)
    return len(placeholders) >= num_placeholders


def check_detectable_content_postscript(response: str, postscript_marker: str, **kwargs) -> bool:
    """Response must contain a postscript section starting with *postscript_marker*."""
    return postscript_marker.lower() in response.lower()


# ---------------------------------------------------------------------------
# Detectable format
# ---------------------------------------------------------------------------


def check_detectable_format_number_bullet_lists(response: str, num_bullets: int, **kwargs) -> bool:
    """Response must contain at least *num_bullets* bullet-list items (lines
    starting with ``-``, ``*``, or ``•``)."""
    bullets = re.findall(r"^\s*[-*•]\s+", response, re.MULTILINE)
    return len(bullets) >= num_bullets


def check_detectable_format_constrained_response(response: str, **kwargs) -> bool:
    """Response must be a single word / short phrase without explanation.
    This check verifies the response is concise (at most 3 words)."""
    return _count_words(response.strip()) <= 3


def check_detectable_format_number_highlighted_sections(response: str, num_highlights: int, **kwargs) -> bool:
    """Response must contain at least *num_highlights* sections highlighted
    with markdown bold (``**text**``) or italic (``*text*``)."""
    highlights = re.findall(r"\*+[^*\n]+\*+", response)
    return len(highlights) >= num_highlights


def check_detectable_format_multiple_sections(
    response: str, section_splitter: str, num_sections: int, **kwargs
) -> bool:
    """Response must contain at least *num_sections* section headers using
    *section_splitter* (e.g. ``###``)."""
    pattern = re.escape(section_splitter)
    headers = re.findall(rf"^{pattern}\s+\S", response, re.MULTILINE)
    return len(headers) >= num_sections


def check_detectable_format_json_format(response: str, **kwargs) -> bool:
    """Response must be valid JSON (possibly wrapped in a markdown code fence)."""
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", response.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def check_detectable_format_title(response: str, **kwargs) -> bool:
    """Response must contain at least one markdown title (``# Title``)."""
    return bool(re.search(r"^#{1,6}\s+\S", response, re.MULTILINE))


# ---------------------------------------------------------------------------
# Combination instructions
# ---------------------------------------------------------------------------


def check_combination_two_responses(response: str, **kwargs) -> bool:
    """Response must contain two parts separated by ``******`` or six or more
    asterisks on a line by themselves."""
    return bool(re.search(r"\*{6,}", response))


def check_combination_repeat_prompt(response: str, prompt_to_repeat: str, **kwargs) -> bool:
    """Response must begin by repeating the prompt verbatim."""
    return response.strip().startswith(prompt_to_repeat.strip())


# ---------------------------------------------------------------------------
# Start / end instructions
# ---------------------------------------------------------------------------


def check_startend_end_checker(response: str, end_phrase: str, **kwargs) -> bool:
    """Response must end with *end_phrase* (case-insensitive, ignoring
    trailing whitespace / punctuation)."""
    cleaned = response.rstrip()
    return cleaned.lower().endswith(end_phrase.lower())


def check_startend_quotation(response: str, **kwargs) -> bool:
    """Response must be wrapped in double quotation marks."""
    stripped = response.strip()
    return stripped.startswith('"') and stripped.endswith('"')


# ---------------------------------------------------------------------------
# Change-case instructions
# ---------------------------------------------------------------------------


def check_change_case_capital_word_frequency(
    response: str, capital_frequency: int, capital_relation: str, **kwargs
) -> bool:
    """A certain proportion of words must be fully capitalised."""
    words = response.split()
    if not words:
        return False
    capital_words = [w for w in words if w.isupper() and w.isalpha()]
    return _compare(len(capital_words), capital_relation, capital_frequency)


def check_change_case_english_capital(response: str, **kwargs) -> bool:
    """Entire response must be in uppercase (ignoring non-alpha characters)."""
    alpha_chars = [c for c in response if c.isalpha()]
    return bool(alpha_chars) and all(c.isupper() for c in alpha_chars)


def check_change_case_english_lowercase(response: str, **kwargs) -> bool:
    """Entire response must be in lowercase (ignoring non-alpha characters)."""
    alpha_chars = [c for c in response if c.isalpha()]
    return bool(alpha_chars) and all(c.islower() for c in alpha_chars)


# ---------------------------------------------------------------------------
# Punctuation instructions
# ---------------------------------------------------------------------------


def check_punctuation_no_comma(response: str, **kwargs) -> bool:
    """Response must not contain any comma."""
    return "," not in response


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_CHECKER_MAP = {
    "keywords:existence": check_keywords_existence,
    "keywords:frequency": check_keywords_frequency,
    "keywords:forbidden_words": check_keywords_forbidden_words,
    "keywords:letter_frequency": check_keywords_letter_frequency,
    "language:response_language": check_language_response_language,
    "length_constraints:number_sentences": check_length_number_sentences,
    "length_constraints:number_paragraphs": check_length_number_paragraphs,
    "length_constraints:number_words": check_length_number_words,
    "length_constraints:nth_paragraph_first_word": check_length_nth_paragraph_first_word,
    "detectable_content:number_placeholders": check_detectable_content_number_placeholders,
    "detectable_content:postscript": check_detectable_content_postscript,
    "detectable_format:number_bullet_lists": check_detectable_format_number_bullet_lists,
    "detectable_format:constrained_response": check_detectable_format_constrained_response,
    "detectable_format:number_highlighted_sections": check_detectable_format_number_highlighted_sections,
    "detectable_format:multiple_sections": check_detectable_format_multiple_sections,
    "detectable_format:json_format": check_detectable_format_json_format,
    "detectable_format:title": check_detectable_format_title,
    "combination:two_responses": check_combination_two_responses,
    "combination:repeat_prompt": check_combination_repeat_prompt,
    "startend:end_checker": check_startend_end_checker,
    "startend:quotation": check_startend_quotation,
    "change_case:capital_word_frequency": check_change_case_capital_word_frequency,
    "change_case:english_capital": check_change_case_english_capital,
    "change_case:english_lowercase": check_change_case_english_lowercase,
    "punctuation:no_comma": check_punctuation_no_comma,
}


def check_instruction(instruction_id: str, response: str, kwargs: Optional[dict] = None) -> bool:
    """Return True if *response* satisfies the instruction identified by
    *instruction_id*.

    Parameters
    ----------
    instruction_id:
        Instruction identifier in the form ``category:subcategory``.
    response:
        The model-generated text to evaluate.
    kwargs:
        Keyword arguments specific to the instruction (may be ``None`` or
        an empty dict for instructions that require no parameters).
    """
    if kwargs is None:
        kwargs = {}
    checker = _CHECKER_MAP.get(instruction_id)
    if checker is None:
        # Unknown instruction type – conservatively return True so as not to
        # penalise models for instructions we cannot verify.
        return True
    try:
        return bool(checker(response, **kwargs))
    except Exception:
        return False
