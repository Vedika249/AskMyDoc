from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)


def evaluate_rag(questions: list, answers: list, contexts: list, ground_truths: list) -> dict:
    """
    Evaluate RAG pipeline using RAGAS framework.

    Metrics:
    - Faithfulness: Is answer grounded in retrieved context?
    - Answer Relevancy: Does answer address the question?
    - Context Recall: Were correct chunks retrieved?
    - Context Precision: Were retrieved chunks relevant?

    Args:
        questions: List of user questions
        answers: List of generated answers
        contexts: List of retrieved context lists
        ground_truths: List of expected answers

    Returns:
        Dict with metric scores
    """
    # Prepare dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    # Run evaluation
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision
        ]
    )

    return results
