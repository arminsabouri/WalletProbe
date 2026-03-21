#! /usr/bin/env python3

import openai
import json
import os
import argparse

from auto_fingerprint.utils import create_embeddings_index
from auto_fingerprint.response import ResponseCollector
from auto_fingerprint.vector_db import QuadrantClient
from auto_fingerprint.source import SourceCodeParser
from auto_fingerprint.exporter import export_fingerprints

def build_parser():
    parser = argparse.ArgumentParser(description="Auto-fingerprint wallet source code.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_common_db_arg(subparser):
        subparser.add_argument(
            "--vector_db_uri", help="URI for the vector database", default="http://localhost:6333"
        )

    embed_parser = subparsers.add_parser("embed", help="Embed the source code and store results")
    embed_parser.add_argument(
        "--dir_to_read", required=True, help="Directory containing the source code and manifest.json"
    )
    _add_common_db_arg(embed_parser)

    fingerprint_parser = subparsers.add_parser(
        "fingerprint", help="Fingerprint the source code and store results"
    )
    fingerprint_parser.add_argument(
        "--dir_to_read", required=True, help="Directory containing the source code and manifest.json"
    )
    _add_common_db_arg(fingerprint_parser)

    export_parser = subparsers.add_parser(
        "export", help="Export wallet fingerprints from Qdrant as JSON"
    )
    _add_common_db_arg(export_parser)
    export_parser.add_argument(
        "--output",
        default="fingerprints.json",
        help="Path to write the exported JSON (default: fingerprints.json)",
    )
    export_parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print the JSON output with indentation."
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command in ("embed", "fingerprint"):
        dir_to_read = args.dir_to_read
        vector_db_uri = args.vector_db_uri
        openai_client = openai.OpenAI()

        try:
            with open(os.path.join(dir_to_read, "manifest.json"), "r") as f:
                manifest = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Manifest file not found in {dir_to_read}")

        wallet_tag = f"{manifest['name']}-{manifest['version']}"
        print("Wallet tag: ", wallet_tag)
        db = QuadrantClient(openai_client, vector_db_uri, wallet_tag)
        db.create_collections()

        if args.command == "embed":
            source_code_parser = SourceCodeParser(dir_to_read, manifest)
            for file_name, functions in source_code_parser.get_files(dir_to_read):
                print(f"found {len(functions)} functions in {file_name}")
                chunk_id_to_text = create_embeddings_index(
                    openai_client, functions, file_name, db
                )

                db.upload_points(chunk_id_to_text)
                print("Uploaded points")

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

            # New independent
            response_collector.output_types()
            response_collector.number_of_outputs()
            response_collector.compressed_public_keys()

            # New probabilistic
            response_collector.input_order_smallest_first()
            response_collector.input_order_largest_first()
            response_collector.input_order_oldest_first()
            response_collector.round_fee_indicator()

            # New dependent
            response_collector.change_type_matches_output()
            response_collector.change_type_matches_input()

            # New temporal
            response_collector.spend_unconfirmed()
            response_collector.rbf_replacement()
            response_collector.feerate_estimation_source()

            print(response_collector.responses)

            db.append_collection_metadata(response_collector.responses)

            print("Done fingerprinting and saving results to the vector db")

    if args.command == "export":
        export_data = export_fingerprints(args.vector_db_uri)
        with open(args.output, "w", encoding="utf-8") as f:
            if args.pretty:
                json.dump(export_data, f, indent=2, sort_keys=True)
            else:
                json.dump(export_data, f, separators=(",", ":"))
        print(f"Wrote fingerprint export to {args.output}")

    print("Exiting...")

if __name__ == "__main__":
    main()
