"""Tests for evaluator functions."""

import pytest
from arize_experiment.evaluators.is_number import is_number


def test_is_number_with_integer():
    """Test is_number evaluator with integer input."""
    score, label, explanation = is_number("123")
    assert score == 1
    assert label == "numeric"
    assert "integer" in explanation.lower()


def test_is_number_with_float():
    """Test is_number evaluator with float input."""
    score, label, explanation = is_number("123.45")
    assert score == 1
    assert label == "numeric"
    assert "float" in explanation.lower()


def test_is_number_with_negative():
    """Test is_number evaluator with negative number input."""
    score, label, explanation = is_number("-123")
    assert score == 1
    assert label == "numeric"
    assert "integer" in explanation.lower()


def test_is_number_with_scientific():
    """Test is_number evaluator with scientific notation input."""
    score, label, explanation = is_number("1.23e-4")
    assert score == 1
    assert label == "numeric"
    assert "float" in explanation.lower()


def test_is_number_with_text():
    """Test is_number evaluator with non-numeric text input."""
    score, label, explanation = is_number("abc")
    assert score == 1
    assert label == "non-numeric"
    assert "cannot be converted" in explanation.lower()


def test_is_number_with_mixed():
    """Test is_number evaluator with mixed numeric and text input."""
    score, label, explanation = is_number("123abc")
    assert score == 1
    assert label == "non-numeric"
    assert "cannot be converted" in explanation.lower()


def test_is_number_with_empty():
    """Test is_number evaluator with empty string input."""
    score, label, explanation = is_number("")
    assert score == 1
    assert label == "non-numeric"
    assert "cannot be converted" in explanation.lower()


def test_is_number_with_whitespace():
    """Test is_number evaluator with whitespace-padded numeric input."""
    score, label, explanation = is_number("  123  ")
    assert score == 1
    assert label == "numeric"
    assert "integer" in explanation.lower()
