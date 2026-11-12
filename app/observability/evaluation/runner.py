"""评测编排:跑 golden set → 算指标 → 出报告 → 存历史。

报告设计原则(和样例项目对齐):
- 旧字段 `top1_accuracy` / `top3_accuracy` / `results` / `failed_cases` 原样保留,
  已有的接口和前端不用改;
- 新指标统一放在 `metrics` 块里,每个都带「适用条数」,避免用 0 冒充「没测到」;
- `config` 记录本次跑的融合策略、重排器、向量后端,否则跨时间的报告无法对比;
- 每次跑追加一行到 `data/eval_history.jsonl` 并写 `eval_runs` 表,用于回归趋势。
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.config.settings import settings
from app.observability.evaluation.metrics import aggregate, evaluate_case
from app.observability.evaluation.ragas_evaluator import evaluate_with_ragas, ragas_available
from app.services.knowledge_service import knowledge_service

logger = logging.getLogger(__name__)

DATA_DIR = settings.DATA_DIR
HISTORY_FILE = DATA_DIR / "eval_history.jsonl"


class EvaluationRunner:
    """一次评测的完整流程。"""

    def __init__(
        self,
        db: Session,
        tenant_id: int,
        tenant_name: str = "",
        top_k: int = 3,
        with_judge: bool = False,
        with_ragas: bool = False,
        dataset_path: str | Path | None = None,
    ) -> None:
        self.db = db
        self.tenant_id = tenant_id
        self.tenant_name = tenant_name
        # 评测口径固定 top_k=3,与 settings.RETRIEVER_TOP_K 解耦:
        # 报告要跨时间对比,不能因为检索默认值调整而漂移。
        self.top_k = top_k
        self.with_judge = with_judge
        self.with_ragas = with_ragas
        self.dataset_path = Path(dataset_path or DATA_DIR / "eval_dataset.json")

    # ------------------------------------------------------------------
    def load_cases(self) -> list[dict]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"评测集不存在: {self.dataset_path}")
        with self.dataset_path.open("r", encoding="utf-8") as f:
            cases = json.load(f)
        if not isinstance(cases, list) or not cases:
            raise ValueError("评测集必须是非空 JSON 数组")
        return cases

    def run(self) -> dict:
        started = time.perf_counter()
        cases = self.load_cases()
        rows: list[dict] = []
        judge_rows: list[dict] = []

        for case in cases:
            query = str(case.get("query", ""))
            results = knowledge_service.search(self.db, self.tenant_id, query, top_k=self.top_k)
            row = evaluate_case(results, case, k=self.top_k)

            if self.with_judge:
                verdict = self._judge_case(query, results, case)
                row.update(
                    {
                        "answer": verdict.get("answer", ""),
                        "faithfulness": verdict.get("faithfulness"),
                        "answer_relevancy": verdict.get("answer_relevancy"),
                        "judge_reason": verdict.get("reason", ""),
                        "judge_failed": verdict.get("judge_failed", True),
                    }
                )
                judge_rows.append(row)

            rows.append(row)

        metrics = aggregate(rows)
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "tenant": self.tenant_name,
            "tenant_id": self.tenant_id,
            "total": len(rows),
            "top_k": self.top_k,
            "top1_accuracy": round(
                sum(1 for r in rows if r["top1_category"] == r["expected_category"]) / len(rows), 4
            )
            if rows
            else 0.0,
            "top3_accuracy": round(sum(1 for r in rows if r["hit"]) / len(rows), 4) if rows else 0.0,
            "metrics": metrics,
            "judge": self._aggregate_judge(judge_rows),
            "ragas": self._run_ragas(rows) if self.with_ragas else {
                "available": ragas_available(),
                "enabled": False,
            },
            "config": {
                "fusion_strategy": settings.FUSION_STRATEGY,
                "rrf_k": settings.RRF_K,
                "vector_store": settings.VECTOR_STORE,
                "reranker": settings.RERANKER_PROVIDER,
                "retriever_candidates": settings.RETRIEVER_CANDIDATES,
                "with_judge": self.with_judge,
            },
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "results": rows,
            "failed_cases": [r for r in rows if not r["hit"]],
        }

        self._persist(report)
        return report

    # ------------------------------------------------------------------
    def _judge_case(self, query: str, results: list[dict], case: dict) -> dict:
        """生成答案并打分;没有 LLM key 时明确标 judge_failed。"""
        from app.observability.evaluation.llm_judge import judge_answer, judge_available

        context = "\n\n".join(str(r.get("content", "")) for r in results)
        if not judge_available():
            return {
                "answer": "",
                "faithfulness": None,
                "answer_relevancy": None,
                "reason": "未配置 LLM_API_KEY,跳过判定",
                "judge_failed": True,
            }

        answer = self._generate_answer(query, context)
        verdict = judge_answer(query, context, answer)
        verdict["answer"] = answer
        return verdict

    @staticmethod
    def _generate_answer(query: str, context: str) -> str:
        """基于检索上下文生成答案(仅用于评估忠实度)。"""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            from app.llm.provider import create_chat_model

            llm = create_chat_model(temperature=0)
            prompt = (
                "请仅依据下面的检索上下文回答用户问题。"
                "上下文不足以回答时,直接说「根据现有资料无法回答」,不要编造。\n\n"
                f"【上下文】\n{context[:4000]}\n\n【问题】\n{query}"
            )
            resp = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=query)])
            return str(getattr(resp, "content", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("评估用答案生成失败: %s", exc)
            return ""

    @staticmethod
    def _aggregate_judge(rows: list[dict]) -> dict:
        if not rows:
            return {"enabled": False, "judged": 0, "failed": 0}
        faith = [r["faithfulness"] for r in rows if r.get("faithfulness") is not None]
        relev = [r["answer_relevancy"] for r in rows if r.get("answer_relevancy") is not None]
        return {
            "enabled": True,
            "judged": len(faith),
            "failed": sum(1 for r in rows if r.get("judge_failed")),
            "faithfulness": round(sum(faith) / len(faith), 4) if faith else None,
            "answer_relevancy": round(sum(relev) / len(relev), 4) if relev else None,
        }

    @staticmethod
    def _run_ragas(rows: list[dict]) -> dict:
        questions = [r["query"] for r in rows]
        answers = [r.get("answer", "") for r in rows]
        contexts = [r.get("contents") or [] for r in rows]
        result = evaluate_with_ragas(questions, answers, contexts)
        result["enabled"] = True
        return result

    def _persist(self, report: dict) -> None:
        """写历史文件 + eval_runs 表;失败不影响已生成的报告。"""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            summary = {
                "generated_at": report["generated_at"],
                "tenant": report["tenant"],
                "total": report["total"],
                "top_k": report["top_k"],
                "top1_accuracy": report["top1_accuracy"],
                "top3_accuracy": report["top3_accuracy"],
                "metrics": report["metrics"],
                "config": report["config"],
            }
            with HISTORY_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("评测历史写入失败: %s", exc)

        try:
            from app.db.models import EvalRun

            record = EvalRun(
                tenant_id=self.tenant_id,
                total=report["total"],
                top_k=report["top_k"],
                top1_accuracy=report["top1_accuracy"],
                top3_accuracy=report["top3_accuracy"],
                metrics=report["metrics"],
                config=report["config"],
            )
            self.db.add(record)
            self.db.commit()
        except Exception as exc:  # noqa: BLE001
            self.db.rollback()
            logger.warning("评测记录入库失败: %s", exc)
