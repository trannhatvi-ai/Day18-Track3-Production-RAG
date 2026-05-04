"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        df = result.to_pandas()

        per_question = []
        for _, row in df.iterrows():
            per_question.append(EvalResult(
                question=row.get("user_input", ""),
                answer=row.get("response", ""),
                contexts=row.get("retrieved_contexts", []) if isinstance(row.get("retrieved_contexts"), list) else [row.get("retrieved_contexts", "")],
                ground_truth=row.get("reference", ""),
                faithfulness=float(row.get("faithfulness", 0) or 0),
                answer_relevancy=float(row.get("answer_relevancy", 0) or 0),
                context_precision=float(row.get("context_precision", 0) or 0),
                context_recall=float(row.get("context_recall", 0) or 0),
            ))

        return {
            "faithfulness": float(df["faithfulness"].mean()) if "faithfulness" in df and df["faithfulness"].notna().any() else 0.0,
            "answer_relevancy": float(df["answer_relevancy"].mean()) if "answer_relevancy" in df and df["answer_relevancy"].notna().any() else 0.0,
            "context_precision": float(df["context_precision"].mean()) if "context_precision" in df and df["context_precision"].notna().any() else 0.0,
            "context_recall": float(df["context_recall"].mean()) if "context_recall" in df and df["context_recall"].notna().any() else 0.0,
            "per_question": per_question,
        }
    except ImportError:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    # Compute average score per question
    scored = []
    for r in eval_results:
        avg_score = (r.faithfulness + r.answer_relevancy + r.context_precision + r.context_recall) / 4
        scored.append((avg_score, r))

    # Sort by score ascending, take bottom_n
    scored.sort(key=lambda x: x[0])
    bottom = scored[:bottom_n]

    failures = []
    for avg_score, r in bottom:
        # Find worst metric
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]

        # Diagnosis mapping
        diagnosis = ""
        suggested_fix = ""
        if worst_metric == "faithfulness" and worst_score < 0.85:
            diagnosis = "LLM hallucinating"
            suggested_fix = "Tighten prompt, lower temperature"
        elif worst_metric == "context_recall" and worst_score < 0.75:
            diagnosis = "Missing relevant chunks"
            suggested_fix = "Improve chunking or add BM25"
        elif worst_metric == "context_precision" and worst_score < 0.75:
            diagnosis = "Too many irrelevant chunks"
            suggested_fix = "Add reranking or metadata filter"
        elif worst_metric == "answer_relevancy" and worst_score < 0.80:
            diagnosis = "Answer doesn't match question"
            suggested_fix = "Improve prompt template"
        else:
            diagnosis = "Other issues"
            suggested_fix = "Review pipeline configuration"

        failures.append({
            "question": r.question,
            "worst_metric": worst_metric,
            "score": worst_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
