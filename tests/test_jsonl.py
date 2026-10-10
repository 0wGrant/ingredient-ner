import json

import pytest

from ingredient_ner.jsonl import append_jsonl, read_jsonl, write_jsonl


def test_write_then_read(tmp_path):
    path = tmp_path / "a.jsonl"
    write_jsonl(path, [{"url": "u1", "steps": ["а", "б"]}, {"url": "u2", "steps": []}])
    assert read_jsonl(path) == [{"url": "u1", "steps": ["а", "б"]}, {"url": "u2", "steps": []}]


def test_write_replaces_content(tmp_path):
    path = tmp_path / "a.jsonl"
    write_jsonl(path, [{"a": 1}, {"a": 2}])
    write_jsonl(path, [{"b": 1}])
    assert read_jsonl(path) == [{"b": 1}]


def test_append_one_record_per_line(tmp_path):
    path = tmp_path / "log.jsonl"
    append_jsonl(path, {"a": 1})
    append_jsonl(path, {"b": 2})
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [{"a": 1}, {"b": 2}]


def test_creates_missing_directories(tmp_path):
    append_jsonl(tmp_path / "x" / "log.jsonl", {"a": 1})
    write_jsonl(tmp_path / "y" / "data.jsonl", [{"b": 2}])
    assert read_jsonl(tmp_path / "x" / "log.jsonl") == [{"a": 1}]
    assert read_jsonl(tmp_path / "y" / "data.jsonl") == [{"b": 2}]


def test_cyrillic_is_readable_in_file(tmp_path):
    path = tmp_path / "a.jsonl"
    write_jsonl(path, [{"name": "Омлет"}])
    raw = path.read_text(encoding="utf-8")
    assert "Омлет" in raw and "\\u" not in raw


def test_read_skips_empty_lines(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "nothing.jsonl")
