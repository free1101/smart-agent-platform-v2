"""文档加载器:PDF / Word / Markdown / 纯文本 → 结构化文本块。

设计要点:
- 统一输出 `LoadedDocument`,上层流水线不关心文件格式;
- **保持原始顺序**:PDF 按页、页内按纵坐标排序,避免多栏排版串行;
- **图片不丢**:图片抽出来单独存,正文里留下 `[[IMAGE:n]]` 占位符,
  由 `ImageCaptioner` 生成描述后原位回填,实现「搜文字能出图」。
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_TOKEN = re.compile(r"\[\[IMAGE:(\d+)\]\]")

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt"}


@dataclass
class LoadedBlock:
    """一个逻辑块(通常对应 PDF 的一页 / Word 的一段)。"""

    text: str
    page: int = 0
    order: int = 0
    images: list[bytes] = field(default_factory=list)


@dataclass
class LoadedDocument:
    """加载结果。"""

    blocks: list[LoadedBlock] = field(default_factory=list)
    file_type: str = ""
    filename: str = ""

    @property
    def images(self) -> list[bytes]:
        return [img for block in self.blocks for img in block.images]

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks if block.text.strip())


class BaseLoader(ABC):
    """加载器接口。"""

    file_type: str = "text"
    suffixes: set[str] = set()

    @abstractmethod
    def load(self, path: Path) -> LoadedDocument:
        """读取文件并返回结构化文本块。"""


class PdfLoader(BaseLoader):
    """PDF 加载器:用 pymupdf(fitz)按页抽取文本与图片。"""

    file_type = "pdf"
    suffixes = {".pdf"}

    def load(self, path: Path) -> LoadedDocument:
        try:
            # PyMuPDF >= 1.24 的正式模块名是 pymupdf;fitz 是旧别名,
            # 2.x 起已标记废弃并提示未来移除,这里优先用新名字。
            import pymupdf as fitz
        except ImportError:
            try:
                import fitz  # 兼容旧版 PyMuPDF
            except ImportError as exc:  # 缺依赖时给出明确指引,而不是静默返回空
                raise RuntimeError("解析 PDF 需要 pymupdf,请执行: pip install pymupdf") from exc

        doc = fitz.open(str(path))
        blocks: list[LoadedBlock] = []
        try:
            for page_no, page in enumerate(doc, start=1):
                # block 结构:(x0, y0, x1, y1, text, block_no, block_type)
                raw_blocks = [
                    b for b in page.get_text("blocks") if len(b) >= 7 and b[6] == 0 and str(b[4]).strip()
                ]
                raw_blocks.sort(key=lambda b: (round(float(b[1]), 1), float(b[0])))
                text = "\n".join(str(b[4]).strip() for b in raw_blocks)

                images: list[bytes] = []
                for img_info in page.get_images(full=True):
                    try:
                        pix = doc.extract_image(img_info[0])
                        data = pix.get("image")
                        if data:
                            images.append(data)
                    except Exception as exc:  # noqa: BLE001 - 单张图损坏不该中断整篇
                        logger.warning("PDF 第 %s 页图片提取失败: %s", page_no, exc)

                if images:
                    tokens = " ".join(f"[[IMAGE:{i}]]" for i in range(len(images)))
                    text = f"{text}\n{tokens}".strip()
                if text.strip():
                    blocks.append(
                        LoadedBlock(text=text, page=page_no, order=len(blocks), images=images)
                    )
        finally:
            doc.close()
        return LoadedDocument(blocks=blocks, file_type=self.file_type, filename=path.name)


class DocxLoader(BaseLoader):
    """Word 加载器:段落按顺序拼接,内联图片附在文末。"""

    file_type = "docx"
    suffixes = {".docx"}

    def load(self, path: Path) -> LoadedDocument:
        try:
            import docx
        except ImportError as exc:
            raise RuntimeError("解析 Word 需要 python-docx,请执行: pip install python-docx") from exc

        document = docx.Document(str(path))
        texts = [p.text.strip() for p in document.paragraphs if p.text.strip()]

        images: list[bytes] = []
        try:
            for rel in document.part.rels.values():
                if "image" not in rel.reltype:
                    continue
                blob = rel.target_part.blob
                if blob:
                    images.append(blob)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Word 图片提取失败: %s", exc)

        text = "\n".join(texts)
        if images:
            text += "\n" + " ".join(f"[[IMAGE:{i}]]" for i in range(len(images)))

        blocks = [LoadedBlock(text=text, page=0, order=0, images=images)] if text.strip() else []
        return LoadedDocument(blocks=blocks, file_type=self.file_type, filename=path.name)


class TextLoader(BaseLoader):
    """Markdown / 纯文本加载器(整篇一个块,先尝试 UTF-8 再回退 GBK)。"""

    file_type = "text"
    suffixes = {".md", ".markdown", ".txt"}

    def load(self, path: Path) -> LoadedDocument:
        raw = path.read_bytes()
        for encoding in ("utf-8", "utf-8-sig", "gbk"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("utf-8", errors="ignore")

        # Markdown 的标题/表格符号对 BM25 有噪声,做轻度清洗但保留正文
        text = text.replace("\r\n", "\n").strip()
        blocks = [LoadedBlock(text=text, page=0, order=0)] if text else []
        return LoadedDocument(blocks=blocks, file_type=self.file_type, filename=path.name)


_LOADERS: list[BaseLoader] = [PdfLoader(), DocxLoader(), TextLoader()]


def get_loader(path: str | Path) -> BaseLoader:
    """按后缀选择加载器。"""
    suffix = Path(path).suffix.lower()
    if suffix == ".doc":
        raise RuntimeError("暂不支持旧版 .doc,请先另存为 .docx")
    for loader in _LOADERS:
        if suffix in loader.suffixes:
            return loader
    raise RuntimeError(f"不支持的文件类型: {suffix}(支持 {sorted(SUPPORTED_SUFFIXES)})")


def load_document(path: str | Path) -> LoadedDocument:
    return get_loader(path).load(Path(path))


def split_images(text: str) -> tuple[str, list[int]]:
    """从文本中找出图片占位符序号(用于回填与校验)。"""
    return IMAGE_TOKEN.sub("", text), [int(m) for m in IMAGE_TOKEN.findall(text)]


def fill_image_captions(text: str, captions: list[str]) -> str:
    """把 `[[IMAGE:n]]` 替换成 `【图片描述】...`;缺失的描述保留占位符。"""

    def _replace(match: re.Match) -> str:
        idx = int(match.group(1))
        if 0 <= idx < len(captions) and captions[idx]:
            return f"【图片描述】{captions[idx]}"
        return match.group(0)

    return IMAGE_TOKEN.sub(_replace, text)


def content_hash(text: str) -> str:
    """单块内容指纹:用于跨文件去重(同一段话不重复入库)。"""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()[:32]
