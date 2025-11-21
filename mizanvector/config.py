import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MizanConfig:
    """Configuration for MizanVector framework.

    Values are read from environment variables by default, but can
    also be passed explicitly when constructing stores.
    """

    db_dsn: Optional[str] = field(default=None)
    db_table: str = field(
        default_factory=lambda: os.getenv("MIZANVECTOR_DB_TABLE", "mizan_documents")
    )
    default_dim: int = field(
        default_factory=lambda: int(os.getenv("MIZANVECTOR_DEFAULT_DIM", "384"))
    )
    default_metric: str = field(
        default_factory=lambda: os.getenv("MIZANVECTOR_DEFAULT_METRIC", "mizan")
    )

    def __post_init__(self) -> None:
        if self.db_dsn is None:
            self.db_dsn = os.getenv(
                "MIZANVECTOR_DB_DSN",
                "postgresql://user:password@localhost:5432/mizanvector",
            )
