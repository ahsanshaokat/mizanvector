"""PostgreSQL + pgvector backend for MizanVector."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import psycopg2
import psycopg2.extras

from .config import MizanConfig
from .metrics import cosine_similarity, euclidean_distance, mizan_similarity
from .store_base import SearchResult, VectorStore


class MizanPgVectorStore(VectorStore):
    """Vector store backed by PostgreSQL + pgvector extension."""

    def __init__(
        self,
        dim: int | None = None,
        dsn: str | None = None,
        table_name: str | None = None,
        config: MizanConfig | None = None,
    ) -> None:
        self._config = config or MizanConfig()
        self.dim = dim or self._config.default_dim
        self.dsn = dsn or self._config.db_dsn
        self.table_name = table_name or self._config.db_table

        self._conn = psycopg2.connect(self.dsn)
        self._ensure_extension_and_table()

    def _ensure_extension_and_table(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    metadata JSONB,
                    embedding vector({self.dim})
                );
                """
            )
            self._conn.commit()

    def add_document(
        self,
        content: str,
        embedding: Sequence[float],
        metadata: Dict[str, Any] | None = None,
    ) -> int:
        vec = np.asarray(embedding, dtype=float)
        if vec.shape[0] != self.dim:
            raise ValueError(
                f"Embedding dim {vec.shape[0]} does not match store dim {self.dim}"
            )
        emb_str = "[" + ",".join(str(float(x)) for x in vec) + "]"
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.table_name} (content, metadata, embedding)
                VALUES (%s, %s, %s)
                RETURNING id;
                """,
                (content, psycopg2.extras.Json(metadata or {}), emb_str),
            )
            doc_id = cur.fetchone()[0]
            self._conn.commit()
        return int(doc_id)

    def _row_to_doc(self, row: Any) -> Dict[str, Any]:
        doc_id, content, metadata, emb = row
        import numpy as np
        if isinstance(emb, str):
            vec = np.fromstring(emb.strip("[]"), sep=",")
        else:
            vec = np.asarray(emb, dtype=float)
        return {
            "id": doc_id,
            "content": content,
            "metadata": metadata or {},
            "embedding": vec,
        }

    def _score(self, q: np.ndarray, v: np.ndarray, metric: str) -> float:
        if metric == "mizan":
            return float(mizan_similarity(q, v))
        if metric == "cosine":
            return float(cosine_similarity(q, v))
        if metric == "l2":
            return -float(euclidean_distance(q, v))
        raise ValueError(f"Unknown metric: {metric}")

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        metric: str = "mizan",
    ) -> List[SearchResult]:
        import numpy as np

        q_vec = np.asarray(query_embedding, dtype=float)
        emb_str = "[" + ",".join(str(float(x)) for x in q_vec) + "]"

        candidates: List[Dict[str, Any]] = []
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, content, metadata, embedding
                FROM {self.table_name}
                ORDER BY embedding <-> %s
                LIMIT %s;
                """,
                (emb_str, top_k * 3),
            )
            rows = cur.fetchall()

        for row in rows:
            candidates.append(self._row_to_doc(row))

        results: List[SearchResult] = []
        for doc in candidates:
            s = self._score(q_vec, doc["embedding"], metric)
            results.append(
                SearchResult(
                    id=doc["id"],
                    score=s,
                    content=doc["content"],
                    metadata=doc["metadata"],
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
