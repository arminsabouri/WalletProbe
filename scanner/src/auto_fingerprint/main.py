#! /usr/bin/env python3

import openai
import numpy as np
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Tree, Node
from pathlib import Path
from typing import Generator
import sys
import json
import os
import lmdb

from auto_fingerprint.utils import create_embeddings_index
from auto_fingerprint.response import ResponseCollector
from auto_fingerprint.vector_db import QuadrantClient

PY_LANGUAGE = Language(tspython.language())
JAVA_LANGUAGE = Language(tsjava.language())


class SourceCodeParser:
    def __init__(self, base_dir: str, manifest: dict):
        self.base_dir = base_dir
        self.source_core_file_suffix = manifest["source_core_file_suffix"]
        if manifest["language"] == "python":
            self.parser = self.__python_parser()
        elif manifest["language"] == "java":
            self.parser = self.__java_parser()
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

    def __java_parser(self) -> Parser:
        parser = Parser(JAVA_LANGUAGE)
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
            lambda node: node.type == "function_definition" or node.type == "method_declaration", self.__traverse_tree(tree))
        functions = list(
            map(lambda node: code[node.start_byte:node.end_byte], function_nodes))

        return functions


def main():
    args = sys.argv[1:]
    if len(args) < 1:
        print("Usage: python main.py <file> <vector_db_uri> <fingerprint_db_path>")
        sys.exit(1)
    dir_to_read = args[0]
    vector_db_uri = args[1]
    fingerprint_db_path = args[2]
    # Initialize the openai client
    openai_client = openai.OpenAI()

    # Read the manifest file
    manifest = None
    try:
        with open(os.path.join(dir_to_read, "manifest.json"), "r") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Manifest file not found in {dir_to_read}")

    wallet_tag = f"{manifest['name']}:{manifest['version']}"
    # Initialize the db
    db = QuadrantClient(openai_client, vector_db_uri, wallet_tag)
    db.create_collections()

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
    response_collector.input_types()
    response_collector.low_r_grinding()
    response_collector.address_reuse()
    response_collector.use_of_nlocktime()
    response_collector.nsequence_value()

    print(response_collector.responses)

    # Save the results to the lmdb database
    env = lmdb.open(fingerprint_db_path, map_size=10**9)  # 1 GB
    response_collector.save_results(env, manifest["name"], manifest["version"])


if __name__ == "__main__":
    main()
