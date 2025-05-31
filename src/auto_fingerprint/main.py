#! /usr/bin/env python3

import openai
import numpy as np
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Tree, Node
from pathlib import Path
from typing import Generator
import sys
import json
import os

from qdrant_client import QdrantClient 
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
import hashlib


PY_LANGUAGE = Language(tspython.language())
VECTOR_DIMENSION = 1536

class SourceCodeParser:
    def __init__(self, base_dir: str, manifest: dict):
        self.base_dir = base_dir
        self.source_core_file_suffix = manifest["source_core_file_suffix"]
        if manifest["language"] == "python":
            self.parser = self.__python_parser()
        else:
            raise ValueError(f"Unsupported language: {manifest['language']}")
    
   
    def get_files(self, base_dir) -> Generator[tuple[str, list[str]], None, None]:
        for f in list(Path(base_dir).rglob(f"*{self.source_core_file_suffix}")):
            with open(f, "r") as f:
                function_str = f.read()
                yield f.name, self.__extract_functions_from_code(function_str)

    def __python_parser(self) -> Parser:

      parser = Parser(PY_LANGUAGE)
      return parser
                
    def __traverse_tree(self, tree: Tree) -> Generator[Node, None, None]:
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
            
    def __extract_functions_from_code(self, code: str) -> list[str]:
        tree = self.parser.parse(bytes(code, "utf8"))
        
        # Filter for function definitions and map them to their code snippets
        function_nodes = filter(lambda node: node.type == "function_definition", self.__traverse_tree(tree))
        functions = list(map(lambda node: code[node.start_byte:node.end_byte], function_nodes))
        
        return functions

# For bad reasons, qdrant doesnt support hex strings as ids. only UUIDs
# So lets convert the sha256 hash to a UUID
def derive_id(function_str: str) -> str:
    sha2 = hashlib.sha256(function_str.encode('utf-8')).hexdigest()[32:]
    return sha2[0:8] + "-" + sha2[8:12] + "-" + sha2[12:16] + "-" + sha2[16:20] + "-" + sha2[20:]

def get_embedding(text: str):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"  # or text-embedding-3-large
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

def create_embeddings_index(code_chunks: list[str], file_name: str, qdrant_client: QdrantClient) -> dict[str, dict]:
    chunk_id_to_text = {}

    # Embed and index the code
    for _, chunk in enumerate(code_chunks):
        sha2 = derive_id(chunk)
        # if id already exists, skip
        if qdrant_client.point_exists(sha2):
            continue
        
        embedding = get_embedding(chunk)
        chunk_id_to_text[sha2] = {
            "file_name": file_name,
            "function_str": chunk,
            "embedding": embedding
        }
        
    return chunk_id_to_text

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
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python main.py <file>")
        sys.exit(1)
    
    # Initialize the db
    db = QuadrantClient()
    db.create_collections()
    
    dir_to_read = args[0]
    
    # Read the manifest file
    manifest = None
    try:
        with open(os.path.join(dir_to_read, "manifest.json"), "r") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Manifest file not found in {dir_to_read}")
    
    source_code_parser = SourceCodeParser(dir_to_read, manifest)
    
    # Read all the files in the directory and embed them
    for file_name, functions in source_code_parser.get_files(dir_to_read):
        print(f"found {len(functions)} functions in {file_name}")
        chunk_id_to_text = create_embeddings_index(functions, file_name, db)
        
        db.upload_points(chunk_id_to_text)
        print(f"uploaded points")
        
    res = db.query("Transaction creation")
    for result in res: 
        print("-", result.id, result.payload["file_name"], result.payload["function_str"])
        
    print('--------------------------------')
    
    res = db.query("BIP69 sorting")
    for result in res: 
        print("-", result.id, result.payload["file_name"], result.payload["function_str"])


if __name__ == "__main__":
    main()

