#! /usr/bin/env python3

import uuid
import openai
import faiss
import numpy as np
import qdrant_client
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Tree, Node
from pathlib import Path
from typing import Generator

from qdrant_client import QdrantClient 
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
import hashlib


PY_LANGUAGE = Language(tspython.language())

VECTOR_DIMENSION = 1536

def python_parser() -> Parser:
    parser = Parser(PY_LANGUAGE)
    return parser

def get_python_files(base_dir):
    return list(Path(base_dir).rglob("*.py"))

def traverse_tree(tree: Tree) -> Generator[Node, None, None]:
    cursor = tree.walk()

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent():
            break
        
        
# For bad reasons, qdrant doesnt support hex strings as ids. only UUIDs
# So lets convert the sha256 hash to a UUID
def derive_id(function_str: str) -> str:
    sha2 = hashlib.sha256(function_str.encode('utf-8')).hexdigest()[32:]
    return sha2[0:8] + "-" + sha2[8:12] + "-" + sha2[12:16] + "-" + sha2[16:20] + "-" + sha2[20:]

def extract_functions_from_code(code, parser):
    tree = parser.parse(bytes(code, "utf8"))
    
    # Filter for function definitions and map them to their code snippets
    function_nodes = filter(lambda node: node.type == "function_definition", traverse_tree(tree))
    functions = list(map(lambda node: code[node.start_byte:node.end_byte], function_nodes))
    
    return functions

def get_embedding(text: str):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"  # or text-embedding-3-large
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

def create_embeddings_index(code_chunks: list[str], file_name: str, qdrant_client: QdrantClient) -> tuple[faiss.IndexFlatL2, dict[str, dict]]:
    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    chunk_id_to_text = {}

    # Embed and index the code
    for _, chunk in enumerate(code_chunks):
        sha2 = derive_id(chunk)
        # if id already exists, skip
        if qdrant_client.point_exists(sha2):
            continue
        
        embedding = get_embedding(chunk)
        index.add(np.array([embedding]))
        chunk_id_to_text[sha2] = {
            "file_name": file_name,
            "function_str": chunk,
            "embedding": embedding
        }
        
    return index, chunk_id_to_text

class QuadrantClient:
    collection_name = "electrum_code_chunks"
    def __init__(self):
        self.qdrant_client = QdrantClient(path="./data.qdrant", prefer_grpc=False)
        
    def create_collections(self) -> None:
        if not self.qdrant_client.get_collection(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE),
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
                    payload={"file_name": item["file_name"], "function_str": item["function_str"]}
                )
            )
       
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
    def query(self, query: str) -> list[dict]:
        query_vec = get_embedding(query)
        res = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=10
        )
        return res
   
def main():
    parser = python_parser()
    db = QuadrantClient()
    db.create_collections()
    files = get_python_files("./test_source_code")
    for file in files:
        with open(file, "r") as f:
            code = f.read()
        functions = extract_functions_from_code(code, parser)
        print(f"found {len(functions)} functions in {file}")
        index, chunk_id_to_text = create_embeddings_index(functions, file.name, db)
        
        db.upload_points(chunk_id_to_text)
        print(f"uploaded points")
        
    res = db.query("Functions related to transaction creation")
    for result in res: 
        print("-", result.payload["file_name"], result.payload["function_str"])


if __name__ == "__main__":
    main()

