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
OPEN_AI_MODEL = "gpt-4o-mini"


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
        function_nodes = filter(
            lambda node: node.type == "function_definition", self.__traverse_tree(tree))
        functions = list(
            map(lambda node: code[node.start_byte:node.end_byte], function_nodes))

        return functions

# For bad reasons, qdrant doesnt support hex strings as ids. only UUIDs
# So lets convert the sha256 hash to a UUID


def derive_id(function_str: str) -> str:
    sha2 = hashlib.sha256(function_str.encode('utf-8')).hexdigest()[32:]
    return sha2[0:8] + "-" + sha2[8:12] + "-" + sha2[12:16] + "-" + sha2[16:20] + "-" + sha2[20:]


def get_embedding(client: openai.OpenAI, text: str):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"  # or text-embedding-3-large
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def create_embeddings_index(client: openai.OpenAI, code_chunks: list[str], file_name: str, qdrant_client: QdrantClient) -> dict[str, dict]:
    chunk_id_to_text = {}

    # Embed and index the code
    for _, chunk in enumerate(code_chunks):
        sha2 = derive_id(chunk)
        # if id already exists, skip
        if qdrant_client.point_exists(sha2):
            continue

        embedding = get_embedding(client, chunk)
        chunk_id_to_text[sha2] = {
            "file_name": file_name,
            "function_str": chunk,
            "embedding": embedding
        }

    return chunk_id_to_text


class QuadrantClient:
    collection_name = "electrum_code_chunks"

    def __init__(self, client: openai.OpenAI):
        self.client = client
        self.qdrant_client = QdrantClient(
            path="./data.qdrant", prefer_grpc=False)

    def create_collections(self) -> None:
        if not self.qdrant_client.get_collection(self.collection_name):
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


class ResponseCollector:
    def __init__(self, vector_db: QuadrantClient, llm: openai.OpenAI):
        self.vector_db = vector_db
        self.llm = llm
        self.responses = {}
        self.chat_history = []

    def save_results(self, wallet_name: str, wallet_version: str):
        key = f"{wallet_name}:{wallet_version}"
        value = json.dumps(self.responses)
        # TODO
        # rocksdb.put(key.encode('utf-8'), value.encode('utf-8'))

    def tx_version(self):
        queries = [
            "transaction version number definition",
            "transaction version initialization",
            "tx version setting"
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results[:3]])
        relevant_chunks = list(set(relevant_chunks))

        system_prompt = """
        You are a code analysis assistant. Your task is to identify the transaction version 
        number used in Bitcoin-related code. Look for:
        - Version numbers in transaction creation
        - Default version values
        - Version constants or definitions
        Only return a single number, or -1 if the version cannot be determined.
        """

        user_prompt = "What transaction version number is used in the following code? Only return a number or -1 if unclear.\n\n"
        user_prompt += "\n\n---\n\n".join(relevant_chunks)

        self._add_to_chat_history("user", user_prompt)

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=16
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE tx version: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["tx_version"] = int(res) if res != "-1" else -1
        except ValueError:
            self.responses["tx_version"] = -1

    def bip69_sorting(self):
        queries = [
            "BIP69 sorting implementation",
            "transaction input output sorting",
            "lexicographical sorting of transactions",
            "BIP69 compliance check"
        ]
        relevant_chunks = []

        # Gather multiple relevant chunks
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results[:3]])

        # Deduplicate chunks
        relevant_chunks = list(set(relevant_chunks))

        system_prompt = """
        You are a code analysis assistant. Your task is to determine if the code implements 
        BIP69 sorting for Bitcoin transactions. Look for:
        - Lexicographical sorting of inputs/outputs
        - References to BIP69 in comments or function names
        - Sorting of transaction inputs by (txid, vout)
        - Sorting of outputs by (amount, scriptPubKey)
        Only return:
        1 - if BIP69 sorting is clearly implemented
        0 - if BIP69 sorting is clearly not implemented
        -1 - if it cannot be determined
        """

        user_prompt = "Does the following code implement BIP69 sorting? Only return 1, 0, or -1 if unclear.\n\n"
        user_prompt += "\n\n---\n\n".join(relevant_chunks)

        self._add_to_chat_history("user", user_prompt)

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=16
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE bip69 sorting: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["bip69_sorting"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["bip69_sorting"] = -1

    def mixed_input_types(self):
        queries = [
            "transaction input type mixing",
            "combine different input types",
            "segwit legacy input combination",
            "transaction input validation",
            "input script type checking"
        ]
        relevant_chunks = []
        for query in queries:
            results = self.vector_db.query(query)
            relevant_chunks.extend([r.payload['function_str']
                                   for r in results[:3]])
        relevant_chunks = list(set(relevant_chunks))

        task_prompt = """
        Analyze the following code to determine if the wallet supports mixing different 
        Bitcoin transaction input types in the same transaction. 
        Look for:
        - Code that handles multiple input types (legacy, segwit, native segwit, taproot)
        - Input type validation or restrictions
        - Transaction building logic that processes different input formats
        - Comments or logic related to input type compatibility
        Only return:
        1 - if mixed input types are clearly supported
        0 - if mixed input types are explicitly prevented
        -1 - if it cannot be determined
        """

        self._add_to_chat_history(
            "user", f"{task_prompt}\n\n" + "\n\n---\n\n".join(relevant_chunks))

        response = self.llm.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=self.chat_history,
            max_tokens=16
        )
        res = response.choices[0].message.content
        print("OPEN AI RESPONSE mixed input types: ", res)

        self._add_to_chat_history("assistant", res)

        try:
            self.responses["mixed_input_types"] = int(
                res) if res in ["0", "1", "-1"] else -1
        except ValueError:
            self.responses["mixed_input_types"] = -1

    def _add_to_chat_history(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content})


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python main.py <file>")
        sys.exit(1)

    # Initialize the openai client
    openai_client = openai.OpenAI()

    # Initialize the db
    db = QuadrantClient(openai_client)
    db.create_collections()

    dir_to_read = args[0]

    # Read the manifest file
    manifest = None
    try:
        with open(os.path.join(dir_to_read, "manifest.json"), "r") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Manifest file not found in {dir_to_read}")

    # Initialize the source code parser
    source_code_parser = SourceCodeParser(dir_to_read, manifest)

    # Read all the files in the directory and embed them
    for file_name, functions in source_code_parser.get_files(dir_to_read):
        print(f"found {len(functions)} functions in {file_name}")
        chunk_id_to_text = create_embeddings_index(
            openai_client, functions, file_name, db)

        db.upload_points(chunk_id_to_text)
        print(f"uploaded points")

    response_collector = ResponseCollector(db, openai_client)
    response_collector.tx_version()
    response_collector.bip69_sorting()
    response_collector.mixed_input_types()
    print(response_collector.responses)


if __name__ == "__main__":
    main()
