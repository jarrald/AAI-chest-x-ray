from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetRunConfig:
    """Identifies one dataset variant in the artifact-removal experiment."""

    name: str
    base_dir: str
    classes: tuple[str, ...]
    source_splits: tuple[str, ...]
    image_exts: tuple[str, ...]
    seed: int


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters that must remain identical across dataset variants."""

    img_size: int
    num_channels: int
    num_epochs: int
    batch_size: int
    lr: float
    dropout: float
    weight_decay: float


@dataclass
class DataBundle:
    """All data objects for one isolated run."""

    config: DatasetRunConfig
    splits_data: dict[str, dict[str, list[str]]]
    train_dataset: Any
    val_dataset: Any
    test_dataset: Any
    train_loader: Any
    val_loader: Any
    test_loader: Any


@dataclass
class ExperimentResult:
    """Training and evaluation outputs for one dataset variant."""

    config: DatasetRunConfig
    training: TrainingConfig
    model: Any
    history: dict[str, list[float]]
    val_threshold: float | None = None
    val_metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)
