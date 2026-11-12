"""摄取流水线测试:Markdown 解析、幂等跳过、块级去重、失败记录。"""

from __future__ import annotations

from pathlib import Path

from app.db.models import IngestionRecord, KnowledgeDoc, TraceRun
from app.ingestion.dedup import file_hash, text_hash
from app.ingestion.loaders import fill_image_captions, load_document, split_images
from app.ingestion.pipeline import IngestionPipeline

SAMPLE = (
    "本店营业时间为每天上午9点到晚上10点,全年无休。\n"
    "肩颈推拿30分钟80元,全身推拿60分钟120元。\n"
    "如需取消预约请提前至少2小时通知,迟到15分钟视为自动取消。\n"
)


def _write(tmp_path: Path, name: str = "guide.md", text: str = SAMPLE) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_text_hash_ignores_whitespace():
    assert text_hash("甲 乙") == text_hash("甲乙")


def test_file_hash_changes_with_content(tmp_path):
    a = _write(tmp_path, "a.md", "内容一")
    b = _write(tmp_path, "b.md", "内容二")
    assert file_hash(a.read_bytes()) != file_hash(b.read_bytes())


def test_load_markdown(tmp_path):
    doc = load_document(_write(tmp_path))
    assert doc.file_type == "text"
    assert "营业时间" in doc.text
    assert doc.images == []


def test_fill_image_captions():
    text = "前文 [[IMAGE:0]] 后文"
    filled = fill_image_captions(text, ["价格表:肩颈80元"])
    assert "【图片描述】价格表:肩颈80元" in filled
    assert "[[IMAGE:0]]" not in filled


def test_fill_image_captions_missing_keeps_token():
    text = "前文 [[IMAGE:0]]"
    assert "[[IMAGE:0]]" in fill_image_captions(text, [""])


def test_split_images():
    _, ids = split_images("a [[IMAGE:0]] b [[IMAGE:1]]")
    assert ids == [0, 1]


def test_pipeline_ingest_and_idempotent(mem_db, tmp_path):
    pipeline = IngestionPipeline()
    path = _write(tmp_path)

    first = pipeline.run(mem_db, tenant_id=1, path=path, with_caption=False)
    assert first.status == "success"
    assert first.chunks >= 1

    stored = mem_db.query(KnowledgeDoc).filter(KnowledgeDoc.tenant_id == 1).all()
    assert len(stored) == first.chunks
    assert stored[0].source_hash == file_hash(path.read_bytes())
    assert stored[0].content_hash

    # 同一文件再跑一次:命中 source_hash,直接跳过
    second = pipeline.run(mem_db, tenant_id=1, path=path, with_caption=False)
    assert second.status == "skipped"
    assert "导入过" in second.skipped_reason


def test_pipeline_block_dedup_on_force(mem_db, tmp_path):
    pipeline = IngestionPipeline()
    path = _write(tmp_path)
    pipeline.run(mem_db, tenant_id=1, path=path, with_caption=False)

    again = pipeline.run(mem_db, tenant_id=1, path=path, with_caption=False, force=True)
    assert again.chunks == 0
    assert again.status == "skipped"


def test_pipeline_records_and_trace(mem_db, tmp_path):
    pipeline = IngestionPipeline()
    result = pipeline.run(mem_db, tenant_id=1, path=_write(tmp_path), with_caption=False)

    record = mem_db.query(IngestionRecord).filter(IngestionRecord.id == result.record_id).one()
    assert record.status == "success"
    assert record.trace_id

    run = mem_db.query(TraceRun).filter(TraceRun.trace_id == result.trace_id).one()
    assert run.kind == "ingestion"
    assert run.status == "success"


def test_pipeline_missing_file(mem_db, tmp_path):
    pipeline = IngestionPipeline()
    result = pipeline.run(mem_db, tenant_id=1, path=tmp_path / "nope.md", with_caption=False)
    assert result.status == "failed"


def test_pipeline_unsupported_suffix(mem_db, tmp_path):
    path = tmp_path / "x.xyz"
    path.write_text("hello", encoding="utf-8")
    result = IngestionPipeline().run(mem_db, tenant_id=1, path=path, with_caption=False)
    assert result.status == "failed"
    assert "不支持" in result.message


def test_pipeline_directory(mem_db, tmp_path):
    for name in ("a.md", "b.md"):
        (tmp_path / name).write_text(SAMPLE + name, encoding="utf-8")
    (tmp_path / "skip.exe").write_bytes(b"\x00")

    results = IngestionPipeline().run_directory(mem_db, 1, tmp_path, with_caption=False)
    assert len(results) == 2
    assert all(r.status == "success" for r in results)
