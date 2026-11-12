"""golden set 校验与统计。

评测集是「回归测试的标尺」,标尺本身错了,后面所有优化都是自欺欺人。
这个脚本做三件事:
1. 结构校验:必填字段、类型、重复 query、空关键词;
2. 分布统计:按分类 / 难度看覆盖度,避免全挤在「easy」;
3. 可选补全:`--fill-difficulty` 给缺 difficulty 的条目补 normal。

用法:
    python scripts/build_golden_set.py
    python scripts/build_golden_set.py --dataset data/eval_dataset.json --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_DATASET = PROJECT_ROOT / "data" / "eval_dataset.json"
REQUIRED = ("query", "expected_category")


def validate(cases: list[dict]) -> list[str]:
    problems: list[str] = []
    seen: dict[str, int] = {}
    for i, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            problems.append(f"[{i}] 不是对象")
            continue
        for field in REQUIRED:
            if not str(case.get(field, "")).strip():
                problems.append(f"[{i}] 缺少必填字段 {field}")
        query = str(case.get("query", "")).strip()
        if query:
            if query in seen:
                problems.append(f"[{i}] query 与第 {seen[query]} 条重复: {query[:30]}")
            else:
                seen[query] = i
        if not case.get("expected_keywords"):
            problems.append(f"[{i}] 缺少 expected_keywords,Recall/ContextPrecision 将无法计算")
        if case.get("difficulty") not in (None, "easy", "medium", "hard"):
            problems.append(f"[{i}] difficulty 取值非法: {case.get('difficulty')}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="golden set 校验")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--strict", action="store_true", help="有问题时以非零码退出")
    parser.add_argument("--fill-difficulty", action="store_true", help="补全缺失的 difficulty 字段")
    args = parser.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        print(f"[错误] 文件不存在: {path}")
        sys.exit(1)

    with path.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    if not isinstance(cases, list):
        print("[错误] 顶层必须是 JSON 数组")
        sys.exit(1)

    problems = validate(cases)
    categories = Counter(str(c.get("expected_category", "")) for c in cases)
    difficulties = Counter(str(c.get("difficulty", "normal")) for c in cases)
    with_keywords = sum(1 for c in cases if c.get("expected_keywords"))
    with_answer = sum(1 for c in cases if c.get("expected_answer"))

    print(f"评测集: {path}")
    print(f"  条目数:            {len(cases)}")
    print(f"  带 expected_keywords: {with_keywords}")
    print(f"  带 expected_answer:   {with_answer}")
    print(f"  分类分布:          {dict(categories)}")
    print(f"  难度分布:          {dict(difficulties)}")

    if problems:
        print(f"\n发现 {len(problems)} 个问题:")
        for item in problems:
            print(f"  - {item}")
    else:
        print("\n校验通过,未发现问题。")

    if args.fill_difficulty:
        changed = 0
        for case in cases:
            if "difficulty" not in case:
                case["difficulty"] = "normal"
                changed += 1
        if changed:
            with path.open("w", encoding="utf-8") as f:
                json.dump(cases, f, ensure_ascii=False, indent=2)
            print(f"\n已补全 {changed} 条 difficulty=normal")

    if problems and args.strict:
        sys.exit(2)


if __name__ == "__main__":
    main()
