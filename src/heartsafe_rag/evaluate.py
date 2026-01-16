"""Evaluation module for assessing RAG system performance against clinical benchmarks.

This module provides functionality to evaluate the RAG system's responses against
a managed Langfuse Dataset using both retrieval and generation components,
with LLM-based judging.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langfuse import Langfuse, observe
from pydantic import BaseModel, Field, ValidationError

from heartsafe_rag.config import settings
from heartsafe_rag.generation import GenerationService
from heartsafe_rag.retrieval import RetrievalService
from heartsafe_rag.utils.logger import logger


class EvaluationError(Exception):
    """Base exception for evaluation-related errors."""

    pass


class DatasetError(EvaluationError):
    """Raised when there's an issue with the evaluation dataset."""

    pass


class EvaluationResult(TypedDict):
    """Type definition for individual evaluation results."""

    id: str
    question: str
    expected: str
    actual: str
    score: float
    reasoning: str


class JudgeVerdict(BaseModel):
    """Schema for the judge's evaluation output."""

    score: float = Field(..., ge=0.0, le=1.0, description="Score from 0 to 1")
    reasoning: str = Field(..., min_length=1, description="Explanation of why the score was given")


class ClinicalJudge:
    """Evaluates clinical responses against expected answers using an LLM judge."""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.0) -> None:
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required for evaluation")

        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=settings.GROQ_API_KEY,
        )

        self.parser = JsonOutputParser(pydantic_object=JudgeVerdict)
        self.prompt = self._create_evaluation_prompt()
        self.chain = self.prompt | self.llm | self.parser

    def _create_evaluation_prompt(self) -> ChatPromptTemplate:
        template = """You are a senior cardiologist evaluating an AI assistant's responses.

        Compare the AI's Actual Answer against the Expected Clinical Answer.

        QUESTION: {question}

        EXPECTED ANSWER (Ground Truth):
        {expected_answer}

        ACTUAL ANSWER (AI Generated):
        {actual_answer}

        ---
        EVALUATION CRITERIA:
        1. **Correctness**: Does the actual answer contain the core clinical facts found in the expected answer?
        2. **Safety**: Does the actual answer contradict the expected answer? (Contradiction = 0 immediately)
        3. **Completeness**: Did it miss critical guidelines mentioned in the expected answer?

        Output valid JSON only:
        {{
            "score": <1 if clinically accurate and safe, 0 if incorrect or unsafe>,
            "reasoning": "<concise explanation>"
        }}
        """
        return ChatPromptTemplate.from_template(template)

    def evaluate(self, question: str, expected: str, actual: str) -> dict[str, Any]:
        if not all([question, expected, actual]):
            raise ValueError("Question, expected, and actual answers must not be empty")

        try:
            return self.chain.invoke({"question": question, "expected_answer": expected, "actual_answer": actual})
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            # Return a fail-safe verdict to avoid crashing the whole run
            return {"score": 0.0, "reasoning": f"Judge Failed: {e!s}"}


async def run_evaluation(
    dataset_name: str = "heartsafe_golden_dataset_v1",
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    delay_seconds: int = 5,
) -> dict[str, Any]:
    """Run the full evaluation pipeline using Langfuse Datasets & Experiments.

    Args:
        dataset_name: Name of the dataset in Langfuse.
        model_name: Name of the LLM model to use for evaluation.
        temperature: Temperature setting for the LLM judge.
        delay_seconds: Delay between evaluations to avoid rate limiting.
    """

    # 1. Initialize Langfuse
    try:
        langfuse = Langfuse()
    except Exception as e:
        raise EvaluationError(f"Failed to initialize Langfuse client: {e}")

    # 2. Fetch Dataset
    try:
        dataset = langfuse.get_dataset(dataset_name)
    except Exception as e:
        raise DatasetError(f"Could not fetch dataset '{dataset_name}'. Have you run upload_dataset.py? Error: {e}")

    # 3. Define Experiment Run Name
    run_name = f"Exp_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    logger.info(f"Starting Experiment: {run_name} on dataset '{dataset_name}'")

    # 4. Initialize Services
    try:
        retrieval_service = RetrievalService()
        generation_service = GenerationService()
        judge = ClinicalJudge(model_name=model_name, temperature=temperature)
    except Exception as e:
        raise EvaluationError(f"Failed to initialize services: {e}")

    results: list[EvaluationResult] = []
    total_score = 0.0

    # 5. Iterate over Dataset Items
    for item in dataset.items:
        # Use item.run() as the context manager for dataset experiments
        with item.run(
            run_name=run_name,
        ) as root_span:
            try:
                q_id = item.id
                question = item.input
                expected = item.expected_output

                # --- A. Retrieval Step ---
                langfuse_handler = root_span.get_langchain_handler()
                docs = []
                category = generation_service.route_query(question, callbacks=[langfuse_handler])

                if category == "HF_RELATED":
                    docs = retrieval_service.retrieve(question, callbacks=[langfuse_handler])

                # --- B. Generation Step ---
                actual_answer = generation_service.generate_response(
                    question,
                    docs,
                    callbacks=[langfuse_handler],
                    category=category,
                )

                # Update trace input and output
                langfuse.update_current_trace(input=question, output=actual_answer)

                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)

                # --- C. Evaluation (Judge) ---
                verdict = judge.evaluate(question, expected, actual_answer)

                # --- D. Scoring (Push to Langfuse UI) ---
                langfuse.create_score(
                    trace_id=root_span.trace_id,
                    name="clinical_accuracy",
                    value=float(verdict["score"]),
                    data_type="NUMERIC",
                    comment=verdict["reasoning"],
                )

                logger.info(f"[{q_id}] Score: {verdict['score']}")

                results.append(
                    {
                        "id": q_id,
                        "question": question,
                        "expected": expected,
                        "actual": actual_answer,
                        "score": float(verdict["score"]),
                        "reasoning": verdict["reasoning"],
                    }
                )
                total_score += float(verdict["score"])

            except Exception as e:
                logger.error(f"Error on item {item.id}: {e}")
                langfuse.update_current_trace(status_message=f"Error: {e!s}")

    # 6. Flush and Report
    langfuse.flush()

    accuracy = (total_score / len(results)) * 100 if results else 0.0
    logger.info(f"Experiment {run_name} finished. Accuracy: {accuracy:.2f}%")
    logger.info("View detailed comparison in Langfuse UI -> Datasets -> Runs")

    return {"run_name": run_name, "accuracy": accuracy, "results": results}


def main() -> None:
    """Main entry point for the evaluation script."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG system using Langfuse Experiments.")
    parser.add_argument(
        "--dataset-name", type=str, default="heartsafe_golden_dataset_v1", help="Name of the dataset in Langfuse"
    )
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile", help="LLM model to use for the Judge")
    parser.add_argument("--delay", type=int, default=5, help="Delay between evaluations in seconds")

    args = parser.parse_args()

    try:
        asyncio.run(run_evaluation(dataset_name=args.dataset_name, model_name=args.model, delay_seconds=args.delay))
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.critical(f"Evaluation failed: {e}", exc_info=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
