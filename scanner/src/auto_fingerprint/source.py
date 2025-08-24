import numpy as np
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser, Tree, Node
from pathlib import Path
from typing import Generator

PY_LANGUAGE = Language(tspython.language())
JAVA_LANGUAGE = Language(tsjava.language())
CPP_LANGUAGE = Language(tscpp.language())


class SourceCodeParser:
    def __init__(self, base_dir: str, manifest: dict):
        self.base_dir = base_dir
        self.source_core_file_suffix = manifest["source_core_file_suffix"]
        if manifest["language"] == "python":
            self.parser = self.__python_parser()
        elif manifest["language"] == "java":
            self.parser = self.__java_parser()
        elif manifest["language"] == "cpp":
            self.parser = self.__cpp_parser()
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

    def __cpp_parser(self) -> Parser:
        parser = Parser(CPP_LANGUAGE)
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
