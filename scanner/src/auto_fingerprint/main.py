#! /usr/bin/env python3

import openai
import json
import os
import argparse

from auto_fingerprint.utils import create_embeddings_index
from auto_fingerprint.response import ResponseCollector
from auto_fingerprint.vector_db import QuadrantClient
from auto_fingerprint.source import SourceCodeParser

def main():
    parser = argparse.ArgumentParser(description="Process source code and vector DB URI.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    def _add_args(parser):
        parser.add_argument("--dir_to_read", help="Directory containing the source code and manifest.json")
        parser.add_argument("--vector_db_uri", help="URI for the vector database")

    embed_parser = subparsers.add_parser("embed", help="Embed the source code and store results")
    fingerprint_parser = subparsers.add_parser("fingerprint", help="Fingerprint the source code and store results")
    _add_args(embed_parser)
    _add_args(fingerprint_parser)

    args = parser.parse_args()

    dir_to_read = args.dir_to_read
    vector_db_uri = args.vector_db_uri
    # Initialize the openai client
    openai_client = openai.OpenAI()

    # Read the manifest file
    manifest = None
    try:
        with open(os.path.join(dir_to_read, "manifest.json"), "r") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Manifest file not found in {dir_to_read}")

    wallet_tag = f"{manifest['name']}-{manifest['version']}"
    print("Wallet tag: ", wallet_tag)
    # Initialize the db
    db = QuadrantClient(openai_client, vector_db_uri, wallet_tag)
    db.create_collections()

    if args.command == "embed":
        # Initialize the source code parser
        source_code_parser = SourceCodeParser(dir_to_read, manifest)
        # Read all the files in the directory and embed them
        for file_name, functions in source_code_parser.get_files(dir_to_read):
            print(f"found {len(functions)} functions in {file_name}")
            chunk_id_to_text = create_embeddings_index(
                openai_client, functions, file_name, db)

            db.upload_points(chunk_id_to_text)
            print(f"Uploaded points")

    if args.command == "fingerprint":
        response_collector = ResponseCollector(db, openai_client)
        response_collector.tx_version()
        response_collector.bip69_sorting()
        response_collector.mixed_input_types()
        response_collector.input_types()
        response_collector.low_r_grinding()
        response_collector.address_reuse()
        response_collector.use_of_nlocktime()
        response_collector.nsequence_value()
        response_collector.change_id_location()
        response_collector.op_return_support()

        print(response_collector.responses)

        # Save results to the vector db
        db.append_collection_metadata(response_collector.responses)

        print("Done fingerprinting and saving results to the vector db")

    print("Exiting...")

if __name__ == "__main__":
    main()
