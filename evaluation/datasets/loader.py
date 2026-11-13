import json
import os
from typing import List, Optional, Dict, Any
from evaluation.datasets.schema import load_jsonl_dataset, validate_dataset, hash_dataset, detect_sample_type
from evaluation.contracts.models import EvalSample


class DatasetLoader:
    """Loader for versioned JSONL datasets."""

    def __init__(self, base_dir: str = "evaluation/datasets"):
        self.base_dir = base_dir

    def load(self, filename: str) -> List[EvalSample]:
        """Load a dataset from a JSONL file."""
        filepath = os.path.join(self.base_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        samples = load_jsonl_dataset(filepath)
        validate_dataset(samples)
        return samples

    def list_datasets(self) -> List[str]:
        """List available dataset filenames."""
        if not os.path.exists(self.base_dir):
            return []
        return [f for f in os.listdir(self.base_dir) if f.endswith(".jsonl")]

    def dataset_hash(self, filename: str) -> str:
        """Compute the hash of a dataset file."""
        filepath = os.path.join(self.base_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        samples = load_jsonl_dataset(filepath)
        return hash_dataset(samples)


class ExperimentDataset:
    """A recorded experiment dataset with versioning."""

    def __init__(
        self,
        samples: List[EvalSample],
        name: str,
        description: str = "",
        version: str = "1.0.0",
    ):
        self.name = name
        self.description = description
        self.version = version
        self.samples = samples
        self.dataset_hash = hash_dataset(samples)
        self.sample_type = detect_sample_type(samples)
        self.created_at = datetime.utcnow().isoformat()

    def to_ragas(self) -> EvaluationDataset:
        """Convert to Ragas EvaluationDataset."""
        from evaluation.datasets.schema import samples_to_ragas_dataset
        return samples_to_ragas_dataset(self.samples)

    def to_jsonl(self, filepath: str) -> None:
        """Export to JSONL file."""
        with open(filepath, "w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(sample_json(sample) + "\n")

    def __len__(self) -> int:
        return len(self.samples)


def sample_json(sample: EvalSample) -> str:
    """Serialize an EvalSample to JSON."""
    return json.dumps(
        {k: v for k, v in sample.__dict__.items() if not k.startswith("_")},
        ensure_ascii=False,
        indent=2,
    )