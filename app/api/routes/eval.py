"""评测 API:运行检索命中率评测并返回报告。

- POST /api/eval/run  执行评测(基于 data/eval_dataset.json)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import require_api_key
from app.api.ratelimit import check_rate_limit
from app.config.settings import settings

router = APIRouter(
    prefix="/api/eval",
    tags=["评测"],
    dependencies=[Depends(require_api_key), Depends(check_rate_limit)],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EVAL_FILE = settings.DATA_DIR / "eval_dataset.json"
REPORT_FILE = settings.DATA_DIR / "eval_report.json"


class EvalRequest(BaseModel):
    tenant_id: int = Field(1, description="租户 id")
    # 评测口径固定 top_k=3,不跟随 RETRIEVER_TOP_K,保证历史报告可对比
    top_k: int = Field(3, ge=1, le=10)
    with_judge: bool = Field(False, description="是否启用 LLM 忠实度/相关性判定(较慢)")


@router.get("/dataset", summary="评测集信息")
def eval_dataset_info():
    if not EVAL_FILE.exists():
        raise HTTPException(404, "评测集不存在")
    with EVAL_FILE.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    return {"total": len(cases), "sample": cases[:3]}


@router.post("/run", summary="运行评测")
def run_eval(req: EvalRequest):
    """调用 scripts/run_eval.py 执行评测,返回报告。"""
    script = PROJECT_ROOT / "scripts" / "run_eval.py"
    if not script.exists():
        raise HTTPException(404, "评测脚本不存在")

    cmd = [
        sys.executable,
        str(script),
        "--tenant", settings.DEFAULT_TENANT_NAME,
        "--top_k", str(req.top_k),
        "--output", str(REPORT_FILE),
    ]
    if req.with_judge:
        cmd.append("--with-judge")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=120, cwd=str(PROJECT_ROOT))
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "评测超时")

    if result.returncode != 0:
        raise HTTPException(500, f"评测失败: {result.stderr[-500:]}")

    if REPORT_FILE.exists():
        with REPORT_FILE.open("r", encoding="utf-8") as f:
            report = json.load(f)
        return report
    return {"detail": result.stdout}
