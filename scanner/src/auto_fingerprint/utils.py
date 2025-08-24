import openai
from auto_fingerprint.vector_db import derive_id, get_embedding
from auto_fingerprint.consts import MAX_TOKENS_FOR_EMBEDDING
from qdrant_client import QdrantClient
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def create_embeddings_index(client: openai.OpenAI, code_chunks: list[str], file_name: str, qdrant_client: QdrantClient) -> dict[str, dict]:
    chunk_id_to_text = {}

    # Embed and index the code
    for _, chunk in enumerate(code_chunks):
        sha2 = derive_id(chunk)
        # if id already exists, skip
        if qdrant_client.point_exists(sha2):
            continue

        if chunk.strip() == "":
            continue

        if num_tokens_from_string(chunk, "gpt-4o-mini") > MAX_TOKENS_FOR_EMBEDDING:
            continue

        embedding = get_embedding(client, chunk)
        chunk_id_to_text[sha2] = {
            "file_name": file_name,
            "function_str": chunk,
            "embedding": embedding
        }

    return chunk_id_to_text

