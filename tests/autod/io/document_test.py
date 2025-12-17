# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json  # noqa: I001
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import benchmark_qed.config.defaults as defs
from benchmark_qed.autod.data_model.document import Document
from benchmark_qed.autod.io.document import (
    create_documents,
    save_documents,
    load_documents,
)
from benchmark_qed.autod.io.enums import InputDataType


def _save_input_docs(
    doc_prefix_path: Path, docs: list[dict[str, Any]], input_data_type: InputDataType
):
    df = pd.DataFrame.from_records(data=docs)
    if input_data_type == InputDataType.PARQUET:
        input_path = doc_prefix_path.with_suffix(".parquet")
        df.to_parquet(input_path)
    elif input_data_type == InputDataType.CSV:
        input_path = doc_prefix_path.with_suffix(".csv")
        df.to_csv(input_path, header=True)
    else:
        msg = f"input_data_type must be {InputDataType.CSV} or {InputDataType.PARQUET}"
        raise ValueError(msg)
    return input_path


def _doc_has_attribute(doc: Document, attr: str) -> bool:
    return doc.attributes is not None and attr in doc.attributes


def _doc_get_attribute(doc: Document, attr: str, default_value: str) -> Any:
    if doc.attributes is not None:
        return doc.attributes.get(attr, default_value)
    return default_value


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


@pytest.mark.parametrize("input_data_type", [InputDataType.CSV, InputDataType.PARQUET])
@pytest.mark.parametrize("file_or_dir", ["file", "dir"])
def test_create_documents_from_dataframe_simple(
    tmp_path: Path, input_data_type: InputDataType, file_or_dir: str
):
    simple_docs = [{"text": "text 1"}, {"text": "text 2"}]

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
    assert docs_sorted_by_title[0].title.endswith(
        "doc" if file_or_dir == "file" else "doc_0"
    )
    assert docs_sorted_by_title[0].text == "text 1"
    assert docs_sorted_by_title[0].type == str(input_data_type)
    assert _doc_has_attribute(docs_sorted_by_title[0], "date_created")
    assert len(docs_sorted_by_title[1].id) > 0
    assert docs_sorted_by_title[1].title.endswith(
        "doc" if file_or_dir == "file" else "doc_1"
    )
    assert docs_sorted_by_title[1].text == "text 2"
    assert docs_sorted_by_title[1].type == str(input_data_type)
    assert _doc_has_attribute(docs_sorted_by_title[1], "date_created")


@pytest.mark.parametrize("input_data_type", [InputDataType.CSV, InputDataType.PARQUET])
@pytest.mark.parametrize("file_or_dir", ["file", "dir"])
def test_create_documents_from_dataframe_complex(
    tmp_path: Path, input_data_type: InputDataType, file_or_dir: str
):
    simple_docs = [
        {
            "content": "text 1",
            "attr1": 1,
            "attr2": "foo",
            "date_created": "20251217T000000Z",
        },
        {
            "content": "text 2truncateme",
            "attr1": 2,
            "attr2": "bar",
            "date_created": "20240101T000000Z",
        },
    ]

    if file_or_dir == "file":
        input_path = _save_input_docs(tmp_path / "doc", simple_docs, input_data_type)
    else:
        for idx, doc in enumerate(simple_docs):
            _save_input_docs(tmp_path / f"doc_{idx}", [doc], input_data_type)
        input_path = tmp_path

    docs = create_documents(
        str(input_path),
        input_type=input_data_type,
        text_tag="content",
        metadata_tags=["attr1", "date_created"],
        max_text_length=6,
    )
    assert len(docs) == 2

    # verify doc title and contents
    docs_sorted_by_title = sorted(docs, key=lambda d: (d.title, d.text))
    assert len(docs_sorted_by_title[0].id) > 0
    assert docs_sorted_by_title[0].title.endswith(
        "doc" if file_or_dir == "file" else "doc_0"
    )
    assert docs_sorted_by_title[0].text == "text 1"
    assert docs_sorted_by_title[0].type == str(input_data_type)
    assert (
        _doc_get_attribute(docs_sorted_by_title[0], "date_created", "")
        == "20251217T000000Z"
    )
    assert _doc_get_attribute(docs_sorted_by_title[0], "attr1", "") == 1
    assert not _doc_has_attribute(docs_sorted_by_title[0], "attr2")
    assert len(docs_sorted_by_title[1].id) > 0
    assert docs_sorted_by_title[1].title.endswith(
        "doc" if file_or_dir == "file" else "doc_1"
    )
    assert docs_sorted_by_title[1].text == "text 2"
    assert docs_sorted_by_title[1].type == str(input_data_type)
    assert (
        _doc_get_attribute(docs_sorted_by_title[1], "date_created", "")
        == "20240101T000000Z"
    )
    assert _doc_get_attribute(docs_sorted_by_title[1], "attr1", "") == 2
    assert not _doc_has_attribute(docs_sorted_by_title[1], "attr2")


@pytest.mark.parametrize("file_or_dir", ["file", "dir"])
def test_create_documents_json_simple(tmp_path: Path, file_or_dir: str):
    simple_docs = [{"text": "text 1"}, {"text": "text 2"}]

    if file_or_dir == "file":
        input_path = tmp_path / "doc.json"
        input_path.write_text(json.dumps(simple_docs[0]), encoding="utf-8")
        expected_count = 1
    else:
        for idx, doc in enumerate(simple_docs):
            file_path = tmp_path / f"doc_{idx}.json"
            file_path.write_text(json.dumps(doc), encoding="utf-8")
        input_path = tmp_path
        expected_count = 2

    docs = create_documents(str(input_path), input_type=InputDataType.JSON)
    assert len(docs) == expected_count

    docs_sorted = sorted(docs, key=lambda d: d.text)
    assert docs_sorted[0].text == "text 1"
    assert docs_sorted[0].type == "json"
    assert _doc_has_attribute(docs_sorted[0], "date_created")


@pytest.mark.parametrize("file_or_dir", ["file", "dir"])
def test_create_documents_json_complex(tmp_path: Path, file_or_dir: str):
    if file_or_dir == "file":
        input_path = tmp_path / "doc.json"
        input_path.write_text(
            '{"content": "text 1 truncateme", "attr1": 1, "attr2": "foo", "date_created": "20251217T000000Z"}',
            encoding="utf-8",
        )
        expected_count = 1
    else:
        docs_data = [
            {
                "content": "text 1 truncateme",
                "attr1": 1,
                "attr2": "foo",
                "date_created": "20251217T000000Z",
            },
            {
                "content": "text 2 truncateme",
                "attr1": 2,
                "attr2": "bar",
                "date_created": "20240101T000000Z",
            },
        ]
        for idx, doc in enumerate(docs_data):
            file_path = tmp_path / f"doc_{idx}.json"
            file_path.write_text(json.dumps(doc), encoding="utf-8")
        input_path = tmp_path
        expected_count = 2

    docs = create_documents(
        str(input_path),
        input_type=InputDataType.JSON,
        text_tag="content",
        metadata_tags=["attr1", "date_created"],
        max_text_length=6,
    )
    assert len(docs) == expected_count

    docs_sorted = sorted(docs, key=lambda d: d.text)
    assert docs_sorted[0].text == "text 1"
    assert _doc_get_attribute(docs_sorted[0], "attr1", "") == 1
    assert _doc_get_attribute(docs_sorted[0], "date_created", "") == "20251217T000000Z"
    assert not _doc_has_attribute(docs_sorted[0], "attr2")

    if expected_count > 1:
        assert docs_sorted[1].text == "text 2"
        assert _doc_get_attribute(docs_sorted[1], "attr1", "") == 2
        assert (
            _doc_get_attribute(docs_sorted[1], "date_created", "") == "20240101T000000Z"
        )
        assert not _doc_has_attribute(docs_sorted[1], "attr2")
        assert {d.short_id for d in docs} == {"0", "1"}


def test_create_documents_text_max_length(tmp_path: Path):
    file = tmp_path / "text_doc.txt"
    file.write_text("hello world truncate this", encoding="utf-8")

    docs = create_documents(
        input_path=str(file), input_type=InputDataType.TEXT, max_text_length=11
    )
    assert len(docs) == 1
    assert docs[0].text == "hello world"
    assert docs[0].title.endswith("text_doc")


def test_create_documents_text_dir_nested(tmp_path: Path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    (tmp_path / "doc1.txt").write_text("root doc", encoding="utf-8")
    (subdir / "doc2.txt").write_text("nested doc", encoding="utf-8")

    docs = create_documents(input_path=str(tmp_path), input_type=InputDataType.TEXT)
    assert len(docs) == 2

    texts = {d.text for d in docs}
    assert texts == {"root doc", "nested doc"}
    assert {d.title.split("/")[-1] for d in docs} == {"doc1", "doc2"}
    assert {d.short_id for d in docs} == {"0", "1"}


@pytest.mark.parametrize("output_dir_exists", [True, False])
def test_create_save_and_load_documents(tmp_path: Path, output_dir_exists: bool):
    (tmp_path / "text_doc_1.txt").write_text("doc 1", encoding="utf-8")
    (tmp_path / "text_doc_2.txt").write_text("doc 2", encoding="utf-8")

    docs = create_documents(input_path=str(tmp_path), input_type=InputDataType.TEXT)
    assert len(docs) == 2

    if output_dir_exists:
        expected_path = tmp_path / f"{defs.DOCUMENT_OUTPUT}.parquet"
        assert expected_path.parent.exists()
    else:
        expected_path = tmp_path / "nested" / f"{defs.DOCUMENT_OUTPUT}.parquet"
        assert not expected_path.parent.exists()

    docs_df = save_documents(docs, output_path=str(expected_path.parent))

    assert len(docs_df) == 2
    assert expected_path.exists()

    loaded_docs = load_documents(docs_df, attributes_cols=["date_created"])
    assert len(loaded_docs) == 2
    for original, loaded in zip(
        sorted(docs, key=lambda d: d.id),
        sorted(loaded_docs, key=lambda d: d.id),
        strict=True,
    ):
        assert original.id == loaded.id
        assert original.short_id == loaded.short_id
        assert original.title == loaded.title
        assert original.text == loaded.text
        assert original.type == loaded.type
        assert original.attributes == loaded.attributes


@pytest.mark.parametrize("file_or_dir", ["file", "dir"])
def test_create_documents_unsupported_input_type(tmp_path: Path, file_or_dir: str):
    input_file = tmp_path / "text_doc_1.txt"
    input_file.write_text("doc 1", encoding="utf-8")
    with pytest.raises(ValueError):  # noqa: PT011, PT012
        if file_or_dir == "file":
            create_documents(str(input_file), input_type="goblin")
        else:
            create_documents(str(tmp_path), input_type="goblin")
