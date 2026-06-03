import argparse
import asyncio
from datetime import datetime
from typing import Any, TypedDict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langfuse import get_client
from pydantic import BaseModel, Field

from heartsafe_rag.config import settings
from heartsafe_rag.generation import GenerationService
from heartsafe_rag.retrieval import HybridRetriever
from heartsafe_rag.utils.logger import logger


class EvaluationError(Exception):
    pass


class DatasetError(EvaluationError):
    pass


class EvaluationResult(TypedDict):
    id: str
    question: str
    expected: str
    actual: str
    score: float
    reasoning: str


class JudgeVerdict(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Score from 0 to 1")
    reasoning: str = Field(..., min_length=1, description="Explanation of why the score was given")


class ClinicalJudge:
    def __init__(self, model_name: str = "llama-3.1-8b-instant", temperature: float = 0.0) -> None:
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
        1. **Faithfulness**: Does the actual answer contain only claims supported by the guidelines, without hallucination?
        2. **Correctness**: Does the actual answer contain the core clinical facts found in the expected answer?
        3. **Safety**: Does the actual answer contradict the expected answer? (Contradiction = score near 0)
        4. **Completeness**: Did it miss critical clinical details mentioned in the expected answer?

        Output valid JSON only:
        {{
            "score": <float between 0.0 and 1.0, where 1.0 is perfectly faithful, correct, and complete>,
            "reasoning": "<concise explanation covering each criterion>"
        }}
        """
        return ChatPromptTemplate.from_template(template)

    def evaluate(self, question: str, expected: str, actual: str) -> dict[str, Any]:
        if not all([question, expected, actual]):
            raise ValueError("Question, expected, and actual answers must not be empty")

        try:
            return self.chain.invoke({  # type: ignore[no-any-return]
                "question": question,
                "expected_answer": expected,
                "actual_answer": actual,
            })
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return {"score": 0.0, "reasoning": f"Judge Failed: {e!s}"}


async def run_evaluation(  # noqa: PLR0912 PLR0915
    dataset_name: str = "heartsafe_golden_dataset_v1",
    model_name: str = "llama-3.1-8b-instant",
    temperature: float = 0.0,
    delay_seconds: int = 5,
) -> dict[str, Any]:
    try:
        langfuse = get_client()
    except Exception as e:
        raise EvaluationError(f"Failed to initialize Langfuse client: {e}") from e

    try:
        dataset = langfuse.get_dataset(dataset_name)
    except Exception as e:
        raise DatasetError(
            f"Could not fetch dataset '{dataset_name}'. Have you run upload_dataset.py? Error: {e}"
        ) from e

    run_name = f"Exp_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    logger.info(f"Starting Experiment: {run_name} on dataset '{dataset_name}'")

    try:
        retrieval_service = HybridRetriever()
        generation_service = GenerationService()
        judge = ClinicalJudge(model_name=model_name, temperature=temperature)
    except Exception as e:
        raise EvaluationError(f"Failed to initialize services: {e}") from e

    results: list[EvaluationResult] = []
    total_score = 0.0

    for item in dataset.items:
        with item.run(
            run_name=run_name,
            run_description="Evaluation run with clinical judge",
            run_metadata={
                "evaluation_run": run_name,
                "model": model_name,
                "temperature": temperature,
                "dataset": dataset_name,
            },
        ) as root_span:
            try:
                q_id = item.id
                question = item.input
                expected = item.expected_output

                langfuse.update_current_trace(input={"query": question})

                with langfuse.start_as_current_observation(
                    name="retrieval", as_type="span"
                ) as retrieval_span:
                    retrieval_span.update(metadata={"item_id": q_id})
                    try:
                        docs = retrieval_service.retrieve(question)

                        formatted_docs = []
                        for i, doc in enumerate(docs):
                            formatted_doc = {
                                "document": {
                                    "content": doc.page_content,
                                    "metadata": doc.metadata,
                                    "source": doc.metadata.get("source", "unknown"),
                                },
                                "rank": i + 1,
                            }
                            formatted_docs.append(formatted_doc)

                        sample_sources = []
                        seen_sources = set()
                        for doc in docs:
                            source = doc.metadata.get("source", "unknown")
                            if source not in seen_sources:
                                seen_sources.add(source)
                                sample_sources.append(source)
                                if len(sample_sources) >= 5:
                                    break

                        retrieval_span.update(
                            input={"query": question},
                            output={
                                "documents": formatted_docs,
                                "retrieved_docs_count": len(docs),
                                "sample_sources": sample_sources,
                                "sample_content": [doc.page_content[:100] + "..." for doc in docs[:3]],
                            },
                        )
                        retrieval_span.update(metadata={"retrieved_docs_count": str(len(docs))})

                    except Exception as e:
                        error_msg = f"Retrieval failed: {e!s}"
                        logger.error(f"Error in retrieval for item {q_id}: {error_msg}")
                        retrieval_span.status_message = error_msg
                        raise EvaluationError(f"Retrieval failed: {e!s}") from e

                with langfuse.start_as_current_observation(
                    name="generation", as_type="span"
                ) as generation_span:
                    generation_span.update(
                        metadata={
                            "item_id": q_id,
                            "model": model_name,
                            "temperature": str(temperature),
                            "context_docs_count": str(len(docs)),
                        }
                    )
                    try:
                        actual_answer = generation_service.generate_response(question, docs)

                        generation_span.update(
                            input={
                                "query": question,
                                "context_docs_count": len(docs),
                                "truncated_context": [doc.page_content[:100] + "..." for doc in docs[:3]] if docs else [],
                            },
                            output={
                                "response": actual_answer,
                                "response_length": len(actual_answer),
                            },
                        )

                    except Exception as e:
                        error_msg = f"Generation failed: {e!s}"
                        logger.error(f"Error in generation for item {q_id}: {error_msg}")
                        generation_span.status_message = error_msg
                        raise EvaluationError(f"Generation failed: {e!s}") from e

                langfuse.update_current_trace(
                    output={
                        "response": actual_answer,
                        "evaluation": {
                            "expected": expected,
                            "actual": actual_answer,
                            "is_correct": actual_answer.strip().lower() == expected.strip().lower(),
                        },
                    }
                )

                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)

                try:
                    verdict = judge.evaluate(question, expected, actual_answer)

                    score = float(verdict.get("score", 0.0))
                    reasoning = str(verdict.get("reasoning", "No reasoning provided")) if verdict.get("reasoning") is not None else "No reasoning provided"

                    langfuse.create_score(
                        trace_id=root_span.trace_id,
                        name="clinical_accuracy",
                        value=score,
                        data_type="NUMERIC",
                        comment=reasoning,
                        metadata={
                            "model": model_name,
                            "item_id": q_id,
                            "question_length": len(question),
                            "response_length": len(actual_answer),
                        },
                    )
                    logger.info(f"[{q_id}] Score: {score}")

                    results.append({
                        "id": q_id,
                        "question": question,
                        "expected": expected,
                        "actual": actual_answer,
                        "score": score,
                        "reasoning": reasoning,
                    })
                    total_score += score
                except Exception as e:
                    error_msg = f"Evaluation failed: {e!s}"
                    logger.error(f"Error in evaluation for item {q_id}: {error_msg}")
                    langfuse.create_score(
                        trace_id=root_span.trace_id,
                        name="evaluation_error",
                        value=0.0,
                        data_type="NUMERIC",
                        comment=error_msg,
                    )
                    raise EvaluationError(error_msg) from e

            except Exception as e:
                error_msg = f"Failed to process item {item.id}: {e!s}"
                logger.error(error_msg, exc_info=True)
                langfuse.update_current_trace(output={"error": error_msg})
                continue

    langfuse.flush()

    accuracy = (total_score / len(results)) * 100 if results else 0.0
    logger.info(f"Experiment {run_name} finished. Accuracy: {accuracy:.2f}%")
    logger.info("View detailed comparison in Langfuse UI -> Datasets -> Runs")

    return {"run_name": run_name, "accuracy": accuracy, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG system using Langfuse Experiments.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="heartsafe_golden_dataset_v1",
        help="Name of the dataset in Langfuse",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-8b-instant",
        help="LLM model to use for the Judge",
    )
    parser.add_argument("--delay", type=int, default=5, help="Delay between evaluations in seconds")

    args = parser.parse_args()

    try:
        asyncio.run(
            run_evaluation(
                dataset_name=args.dataset_name,
                model_name=args.model,
                delay_seconds=args.delay,
            )
        )
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.critical(f"Evaluation failed: {e}", exc_info=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
