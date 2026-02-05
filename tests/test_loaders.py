from pathlib import Path

from rag.loaders import load_files


def test_load_txt_file(tmp_path: Path) -> None:
    txt = tmp_path / "sample.txt"
    txt.write_text("Hello world")

    docs = load_files([txt])

    assert len(docs) == 1
    assert "Hello world" in docs[0].page_content
    assert docs[0].metadata["source"].endswith("sample.txt")
