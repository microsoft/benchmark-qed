# Copyright (c) 2025 Microsoft Corporation.
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from benchmark_qed.autod.io.document import create_documents
from benchmark_qed.autod.io.enums import InputDataType

def test_create_documents_text_file(tmp_path: Path):
    file = tmp_path / "text_doc.txt"
    text = "here is a text document"
    file.write_text(text, encoding="utf-8")

    docs = create_documents(input_path=str(file), input_type=InputDataType.TEXT)
    assert len(docs) == 1
    assert docs[0].text == text
    assert docs[0].title.endswith("text_doc")

def test_create_documents_text_dir(tmp_path: Path):
    file_1 = tmp_path / "text_doc_1.txt"
    text_1 = "1"
    file_1.write_text(text_1, encoding="utf-8")

    file_2 = tmp_path / "text_doc_2.txt"
    text_2 = "2"
    file_2.write_text(text_2, encoding="utf-8")

    docs = create_documents(input_path=str(tmp_path), input_type=InputDataType.TEXT)
    assert len(docs) == 2

    # verify doc title and contents
    docs_sorted_by_title = sorted(docs, key=lambda d: d.title)
    assert docs_sorted_by_title[0].title.endswith("text_doc_1")
    assert docs_sorted_by_title[0].text == text_1
    assert docs_sorted_by_title[1].title.endswith("text_doc_2")
    assert docs_sorted_by_title[1].text == text_2


def _save_input_docs(doc_prefix_path: Path, docs: list[dict[str, Any]], input_data_type: InputDataType):
    df = pd.DataFrame.from_records(data=docs)
    if input_data_type == InputDataType.PARQUET:
        input_path = doc_prefix_path.with_suffix(".parquet")
        df.to_parquet(input_path)
    elif input_data_type == InputDataType.CSV:
        input_path = doc_prefix_path.with_suffix(".csv")
        df.to_csv(input_path, header=True)
    return input_path


@pytest.mark.parametrize("input_data_type", [InputDataType.CSV, InputDataType.PARQUET])
@pytest.mark.parametrize("file_or_dir", ["file", "dir"])
def test_create_documents_from_dataframe_simple(tmp_path: Path, input_data_type: InputDataType, file_or_dir: str):
    simple_docs = [
        {"text": "text 1"},
        {"text": "text 2"}
    ]

    if file_or_dir == "file":
        input_path = _save_input_docs(tmp_path / "doc", simple_docs, input_data_type)
    else:
        for idx, doc in enumerate(simple_docs):
             _save_input_docs(tmp_path / f"doc_{idx}", [doc], input_data_type)
        input_path = tmp_path

    docs = create_documents(str(input_path), input_type=input_data_type)
    assert len(docs) == 2

    # verify doc title and contents
    docs_sorted_by_title = sorted(docs, key=lambda d: (d.title, d.text))
    assert len(docs_sorted_by_title[0].id) > 0
    assert docs_sorted_by_title[0].title.endswith("doc" if file_or_dir == "file" else "doc_0")
    assert docs_sorted_by_title[0].text == "text 1"
    assert docs_sorted_by_title[0].type == str(input_data_type)
    assert 'date_created' in docs_sorted_by_title[0].attributes
    assert len(docs_sorted_by_title[1].id) > 0
    assert docs_sorted_by_title[1].title.endswith("doc" if file_or_dir == "file" else "doc_1")
    assert docs_sorted_by_title[1].text == "text 2"
    assert docs_sorted_by_title[1].type == str(input_data_type)
    assert 'date_created' in docs_sorted_by_title[1].attributes


@pytest.mark.parametrize("input_data_type", [InputDataType.CSV, InputDataType.PARQUET])
@pytest.mark.parametrize("file_or_dir", ["file", "dir"])
def test_create_documents_from_dataframe_complex(tmp_path: Path, input_data_type, file_or_dir: str):
    # save a test parquet file
    simple_docs = [
        {"content": "text 1", "attr1": 1, "attr2": "foo", "date_created": "20251217T000000Z"},
        {"content": "text 2truncateme", "attr1": 2, "attr2": "bar", "date_created": "20240101T000000Z"}
    ]

    if file_or_dir == "file":
        input_path = _save_input_docs(tmp_path / "doc", simple_docs, input_data_type)
    else:
        for idx, doc in enumerate(simple_docs):
            _save_input_docs(tmp_path / f"doc_{idx}", [doc], input_data_type)
        input_path = tmp_path

    docs = create_documents(str(input_path), input_type=input_data_type, text_tag="content", metadata_tags=["attr1", "date_created"], max_text_length=6)
    assert len(docs) == 2

    # verify doc title and contents
    docs_sorted_by_title = sorted(docs, key=lambda d: (d.title, d.text))
    assert len(docs_sorted_by_title[0].id) > 0
    assert docs_sorted_by_title[0].title.endswith("doc" if file_or_dir == "file" else "doc_0")
    assert docs_sorted_by_title[0].text == "text 1"
    assert docs_sorted_by_title[0].type == str(input_data_type)
    assert docs_sorted_by_title[0].attributes["date_created"] == "20251217T000000Z"
    assert docs_sorted_by_title[0].attributes["attr1"] == 1
    assert "attr2" not in docs_sorted_by_title[0].attributes
    assert len(docs_sorted_by_title[1].id) > 0
    assert docs_sorted_by_title[1].title.endswith("doc" if file_or_dir == "file" else "doc_1")
    assert docs_sorted_by_title[1].text == "text 2"
    assert docs_sorted_by_title[1].type == str(input_data_type)
    assert docs_sorted_by_title[1].attributes["date_created"] == "20240101T000000Z"
    assert docs_sorted_by_title[1].attributes["attr1"] == 2
    assert "attr2" not in docs_sorted_by_title[1].attributes
