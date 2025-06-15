import hashlib
import openai
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from auto_fingerprint.consts import VECTOR_DIMENSION, EMBEDDING_MODEL

# For bad reasons, qdrant doesnt support hex strings as ids. only UUIDs
# So lets convert the sha256 hash to a UUID
def derive_id(function_str: str) -> str:
    sha2 = hashlib.sha256(function_str.encode('utf-8')).hexdigest()[32:]
    return sha2[0:8] + "-" + sha2[8:12] + "-" + sha2[12:16] + "-" + sha2[16:20] + "-" + sha2[20:]


def get_embedding(client: openai.OpenAI, text: str):
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


class QuadrantClient:
    collection_name = "electrum_code_chunks"

    def __init__(self, client: openai.OpenAI, vector_db_uri: str):
        self.client = client
        self.qdrant_client = QdrantClient(
            url=vector_db_uri)

    def create_collections(self) -> None:
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_DIMENSION, distance=Distance.COSINE),
            )

    def point_exists(self, id: str) -> bool:
        results = self.qdrant_client.retrieve(
            collection_name=self.collection_name,
            ids=[id]
        )
        return len(results) > 0

    def upload_points(self, embeddings: dict[str, dict]) -> None:
        points = []
        for sha2, item in embeddings.items():
            points.append(
                PointStruct(
                    id=sha2,
                    vector=item["embedding"].tolist(),
                    payload={"file_name": item["file_name"],
                             "function_str": item["function_str"]}
                )
            )

        if len(points) > 0:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )

    def query(self, query: str) -> list[dict]:
        query_vec = get_embedding(self.client, query)
        res = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=10
        )
        return res

