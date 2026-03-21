from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient

from auto_fingerprint.heuristics import HEURISTICS
from auto_fingerprint.vector_db import derive_id


def _get_metadata(db: QdrantClient, collection_name: str, metadata_id: str) -> dict:
    """Fetch the metadata point payload for a wallet collection."""
    try:
        res = db.retrieve(
            collection_name=collection_name,
            ids=[metadata_id],
            with_payload=True,
            with_vectors=False,
        )
        if res:
            return res[0].payload or {}
    except Exception as exc:
        return {"error": f"could not load metadata: {exc}"}
    return {}


def export_fingerprints(vector_db_uri: str) -> dict[str, Any]:
    """Pull all wallet metadata points from Qdrant and return a serializable dict."""
    db = QdrantClient(url=vector_db_uri, prefer_grpc=False)
    collections = db.get_collections().collections
    metadata_id = derive_id("metadata")

    wallets = []
    for col in sorted(collections, key=lambda c: c.name):
        payload = _get_metadata(db, col.name, metadata_id)
        wallets.append({"name": col.name, "fingerprints": payload})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vector_db_uri": vector_db_uri,
        "wallets": wallets,
        "heuristics": HEURISTICS,
    }
