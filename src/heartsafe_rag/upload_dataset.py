import json
from pathlib import Path

from langfuse import Langfuse

from heartsafe_rag.config import settings
from heartsafe_rag.utils.logger import logger


def upload_dataset():
    """
    Uploads the local golden_dataset.json to Langfuse as a managed Dataset.
    """
    # Initialize Langfuse Client (picks up env vars automatically)
    langfuse = Langfuse()

    dataset_name = "heartsafe_golden_dataset_v1"

    # Create or Get the Dataset
    try:
        langfuse.create_dataset(
            name=dataset_name, description="Ground truth questions and answers for Heart Failure Guidelines (AHA 2022)"
        )
        logger.info(f"Dataset '{dataset_name}' created/verified.")
    except Exception as e:
        logger.warning(f"Note regarding dataset creation: {e}")

    # Load Local JSON
    json_path = Path("eval/data/golden_dataset.json")
    if not json_path.exists():
        logger.error(f"File not found: {json_path}")
        return

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Upsert Items
    for item in data:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=item["question"],
            expected_output=item["expected_answer"],
            metadata={"id": item["id"], "source_text": item.get("source", "")},
        )
        logger.info(f"Uploaded item ID: {item['id']}")

    logger.info("Dataset upload complete. Check Langfuse UI.")


if __name__ == "__main__":
    upload_dataset()
