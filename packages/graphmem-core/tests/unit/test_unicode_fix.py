"""Test Unicode escape fixes in LLM output and storage."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphmem.llm.openai_compatible import _fix_bare_unicode_escapes


def test_fix_bare_unicode_escapes():
    raw = ["u65e5u6d3bu5341u4e07u3001QPSu5cf0u503c5000"]
    fixed = _fix_bare_unicode_escapes(raw)
    assert fixed == ["日活十万、QPS峰值5000"]


def test_fix_nested_dict():
    raw = {"key_points": ["u5355u4f53u67b6u6784"], "title": "u6d4bu8bd5"}
    fixed = _fix_bare_unicode_escapes(raw)
    assert fixed == {"key_points": ["单体架构"], "title": "测试"}


def test_does_not_touch_properly_escaped():
    raw = ["\u65e5\u672c"]  # properly escaped
    fixed = _fix_bare_unicode_escapes(raw)
    assert fixed == ["日本"]
