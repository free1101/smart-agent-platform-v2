"""MD Human Review 批注定位辅助工具。

用途：评审指令文件（.review/*_aicmd.md）里的 blockIndex / 文本锚点 有时会互相打架
（面板把批注吸到相邻块）。本工具用 *_record.md 里的 rawMarkdown（批阅时刻的源文快照）
按「空行分割」规则复算 blockIndex，并支持按关键词反查块，用于交叉验证。

用法：
    # 1) 复算块内容（可给多个索引）
    python scripts/review_locate.py .review/tmp_v1_record.md block 43 48 86

    # 2) 列出所有块（带索引，便于人工核对）
    python scripts/review_locate.py .review/tmp_v1_record.md list | more

    # 3) 按关键词反查块（语义定位：找「问题被提出的那一节」）
    python scripts/review_locate.py .review/tmp_v1_record.md find keywords

    # 4) 直接用当前源文件（非快照）复算
    python scripts/review_locate.py tmp.md block 43 --raw

    # 5) 体检：拿 aicmd 指令去核对快照，自动找出锚点错位的批注
    python scripts/review_locate.py check .review/tmp_v2_record.md .review/tmp_1788923332075_aicmd.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# 面板定义：blockIndex 基于空行分割（从 0 开始）
_BLANK_LINE = re.compile(r"\n[ \t]*\n")

# 评论里值得当「定位实体」的模式：反引号标识符 + 长度>=4 的 ASCII 词
_ENTITY = re.compile(r"`([A-Za-z_][\w.]*)`|\b([A-Za-z_][\w]{3,})\b")

# 回指词：出现时说明评论里的实体多半是「参照物」，不是评论主题
_REFERRAL = re.compile(r"前面的|上面的|还是用|刚才|前述|上一")

# 挖坑处标记：原文自己提出却没解答的地方，才是答案真正该落的位置
_HOOKS = ("讲不清", "待补", "还没解释", "先记住", "留个坑", "后面再说", "暂时不解释")


def load_raw(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```json\n(.*?)\n```", text, re.S)
    if match:
        return json.loads(match.group(1))["rawMarkdown"]
    return text


def split_blocks(raw: str) -> list[str]:
    raw = raw.replace("\r\n", "\n")
    return [b.strip("\n") for b in _BLANK_LINE.split(raw)]


def preview(block: str, limit: int = 200) -> str:
    flat = block.replace("\n", " / ")
    return flat if len(flat) <= limit else flat[:limit] + " ..."


def norm(text: str) -> str:
    """归一化：去掉 markdown 强调符与空白，用于容忍锚点/目标文本与源文的格式差异。"""
    return re.sub(r"[\s*`]", "", text)


def parse_aicmd(path: Path) -> list[dict]:
    """从评审指令文件里解析出每条指令的定位块 / 目标文本 / 评论。

    优先取「目标文本」代码块（干净、无反引号）；取不到才退回「文本锚点」。
    """
    text = path.read_text(encoding="utf-8")
    items: list[dict] = []
    for chunk in re.split(r"\n## 指令\s*\d+", text)[1:]:
        block = re.search(r"\*\*定位块\*\*：第\s*(\d+)\s*块", chunk)
        target = re.search(r"\*\*目标文本\*\*：\s*```\n(.*?)\n```", chunk, re.S)
        anchor = re.search(r"\*\*文本锚点\*\*：`(.*?)`\s*\n", chunk, re.S)
        comment = re.search(r"\*\*评论内容\*\*：(.*)", chunk)
        if not (block and comment):
            continue
        raw_target = (target.group(1) if target else (anchor.group(1) if anchor else "")).strip()
        items.append(
            {
                "block": int(block.group(1)),
                "target": raw_target,
                "comment": comment.group(1).strip(),
            }
        )
    return items


def entities_of(comment: str) -> list[str]:
    found = set()
    for backticked, bare in _ENTITY.findall(comment):
        token = backticked or bare
        if len(token) >= 4:
            found.add(token)
    return sorted(found)


def check(record: Path, aicmd: Path) -> None:
    """核对指令锚点是否可信：锚点能否在文中找到、评论实体是否出现在目标块。"""
    blocks = split_blocks(load_raw(record))
    print(f"快照 {record.name} 共 {len(blocks)} 块\n")
    for i, item in enumerate(parse_aicmd(aicmd), 1):
        key = item["target"].splitlines()[0].strip().lstrip("- ").strip()
        nkey = norm(key)
        hits = [j for j, b in enumerate(blocks) if nkey and nkey in norm(b)]
        claimed = item["block"]
        located = hits[0] if hits else None
        print(f"指令 {i}  声称 block[{claimed}]  实际锚点命中 {hits or '未找到'}")

        if located is None:
            print("  [!] 锚点在快照中不存在 —— 可能快照版本不对，或锚点被截断")
            continue
        if located != claimed:
            print(f"  [i] blockIndex 与空行分割索引存在系统偏移 {claimed - located:+d}（不影响定位，以锚点为准）")

        target = norm(blocks[located])
        missing = [e for e in entities_of(item["comment"]) if norm(e) not in target]
        if not missing:
            print("  [OK] 评论实体均出现在目标块，定位可信")
            continue

        print(f"  [!] 评论实体 {missing} 不在目标块 —— 锚点可疑")
        if _REFERRAL.search(item["comment"]):
            print("      [i] 评论含回指词（前面的/还是用/…），实体更可能是「参照物」而非主题，谨慎搬移")

        # 搬移的充分条件＝候选块里存在「挖坑处」，而不是实体缺失本身
        raised: list[int] = []
        for e in missing:
            for j, b in enumerate(blocks):
                if norm(e) not in norm(b):
                    continue
                hook = next((w for w in _HOOKS if w in b), None)
                tag = f" <== 原文在此提出该问题（{hook}）" if hook else ""
                print(f"      {e} -> block[{j}] {preview(b, 120)}{tag}")
                if hook and j != located and j not in raised:
                    raised.append(j)

        if raised:
            print(f"      ==> 建议改落到 block[{raised[0]}]（原文挖坑处）")
        else:
            print("      ==> 候选块无挖坑信号 —— 保持锚点位置，不要因为实体缺失就搬走")
        print()


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return

    # Windows 控制台默认 GBK，含 emoji/特殊符号会崩，强制 UTF-8 输出
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if sys.argv[1] == "check":
        check(Path(sys.argv[2]), Path(sys.argv[3]))
        return

    path = Path(sys.argv[1])
    mode = sys.argv[2]
    args = [a for a in sys.argv[3:] if not a.startswith("--")]
    blocks = split_blocks(load_raw(path))

    if mode == "list":
        for i, b in enumerate(blocks):
            print(f"[{i}] {preview(b, 160)}")
        print(f"\n共 {len(blocks)} 块")
        return

    if mode == "block":
        for a in args:
            i = int(a)
            if not 0 <= i < len(blocks):
                print(f"[{i}] 越界（共 {len(blocks)} 块）")
                continue
            print(f"--- block[{i}] ---")
            print(blocks[i])
            print()
        return

    if mode == "find":
        kw = args[0] if args else ""
        hits = [(i, b) for i, b in enumerate(blocks) if kw in b]
        if not hits:
            print(f"未找到含「{kw}」的块")
            return
        for i, b in hits:
            print(f"[{i}] {preview(b, 200)}")
        print(f"\n命中 {len(hits)} 块；首次出现在 block[{hits[0][0]}]")
        return

    print(__doc__)


if __name__ == "__main__":
    main()
