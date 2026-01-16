import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, dict, list

# Mocking the import for the script - in real life import your actual RAG chain
# from src.rag import generate_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path("data/golden_dataset.json")
OUTPUT_PATH = Path("eval_results.json")


@dataclass
class EvalResult:
    question_id: str
    question: str
    ground_truth: str
    generated_answer: str
    is_correct: bool
    latency_ms: float = 0.0


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def llm_judge(generated: str, truth: str) -> bool:
    """
    Simulates an LLM-as-a-Judge.
    In production, this would call GPT-4/Llama-70b to compare meaning.
    Here, we use simple string presence for the demo.
    """
    if truth == "REFUSAL_SIGNAL":
        # Check if the model refused to answer (safety check)
        refusal_keywords = ["cannot answer", "guidelines do not", "out of scope", "apologize"]
        return any(keyword in generated.lower() for keyword in refusal_keywords)

    # Simple keyword heuristic for correctness (Replace with LLM call in v2)
    # This is a 'Naive' evaluator for demonstration
    truth_keywords = [w for w in truth.lower().split() if len(w) > 3]
    match_count = sum(1 for w in truth_keywords if w in generated.lower())
    return match_count / len(truth_keywords) > 0.5 if truth_keywords else False


def run_evaluation():
    logger.info("Starting Evaluation Pipeline...")
    dataset = load_dataset(DATASET_PATH)
    results = []
    correct_count = 0

    print(f"{'ID':<6} | {'Status':<10} | {'Question Summary'}")
    print("-" * 50)

    for item in dataset:
        # 1. Inference (Replace with actual RAG call)
        # generated_answer = generate_answer(item['question'])

        # MOCK RESPONSE for Demo purposes:
        if item["ground_truth"] == "REFUSAL_SIGNAL":
            generated_answer = "I cannot answer this as it is outside the HF guidelines."
        else:
            generated_answer = item["ground_truth"] + " (Cited: Guideline 2022)"

        # 2. Evaluation (LLM-as-a-Judge)
        is_correct = llm_judge(generated_answer, item["ground_truth"])

        if is_correct:
            correct_count += 1

        results.append(
            EvalResult(
                question_id=item["id"],
                question=item["question"],
                ground_truth=item["ground_truth"],
                generated_answer=generated_answer,
                is_correct=is_correct,
            )
        )

        status_icon = "✅ PASS" if is_correct else "❌ FAIL"
        print(f"{item['id']:<6} | {status_icon:<10} | {item['question'][:30]}...")

    score = (correct_count / len(dataset)) * 100
    logger.info(f"Evaluation Complete. Final Accuracy: {score:.2f}%")

    # Save detailed report
    with open(OUTPUT_PATH, "w") as f:
        json.dump([vars(r) for r in results], f, indent=2)


if __name__ == "__main__":
    run_evaluation()
