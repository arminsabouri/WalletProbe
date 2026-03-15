from flask import Flask, render_template, request
import argparse
import hashlib
from qdrant_client import QdrantClient
# import numpy as np
# import openai

app = Flask(__name__)

VECTOR_DIMENSION = 1536
METADATA_VECTOR = [0] * VECTOR_DIMENSION

HEURISTICS = {
    "independent": [
        {"key": "tx_version", "label": "Transaction Version"},
        {"key": "input_types", "label": "Input Types"},
        {"key": "mixed_input_types", "label": "Mixed Input Types"},
        {"key": "output_types", "label": "Output Types"},
        {"key": "number_of_outputs", "label": "Number of Outputs"},
        {"key": "nsequence_value", "label": "nSequence Value (RBF Signaling)"},
        {"key": "compressed_public_keys", "label": "Compressed Public Keys"},
        {"key": "use_of_nlocktime", "label": "Anti-Fee-Sniping (nLockTime)"},
        {"key": "op_return_support", "label": "OP_RETURN Support"},
        {"key": "address_reuse", "label": "Address Reuse"},
        {"key": "low_r_grinding", "label": "Low-R Grinding"},
    ],
    "probabilistic": [
        {"key": "bip69_sorting", "label": "BIP 69 Sorting"},
        {"key": "input_order_smallest_first", "label": "Input Order: Smallest First"},
        {"key": "input_order_largest_first", "label": "Input Order: Largest First"},
        {"key": "input_order_oldest_first", "label": "Input Order: Oldest First"},
        {"key": "round_fee_indicator", "label": "Round Fee Indicator"},
    ],
    "dependent": [
        {"key": "change_id_location", "label": "Change Position in Outputs"},
        {"key": "change_address_same_as_input", "label": "Change Address Same as Input"},
        {"key": "change_type_matches_output", "label": "Change Type Matches Output"},
        {"key": "change_type_matches_input", "label": "Change Type Matches Input"},
    ],
    "temporal": [
        {"key": "spend_unconfirmed", "label": "Spend Unconfirmed"},
        {"key": "rbf_replacement", "label": "RBF Replacement"},
        {"key": "feerate_estimation_source", "label": "Feerate Estimation Source"},
    ],
}


def derive_metadata_id():
    sha2 = hashlib.sha256("metadata".encode('utf-8')).hexdigest()[32:]
    return sha2[0:8] + "-" + sha2[8:12] + "-" + sha2[12:16] + "-" + sha2[16:20] + "-" + sha2[20:]


class VectorDBFactory:
    def __init__(self, vector_db_uri):
        self.vector_db_uri = vector_db_uri

    def db(self):
        return QdrantClient(url=self.vector_db_uri, prefer_grpc=False)


def get_wallet_fingerprints(db, collection_name):
    """Fetch the metadata point payload for a wallet collection."""
    metadata_id = derive_metadata_id()
    try:
        res = db.retrieve(
            collection_name=collection_name,
            ids=[metadata_id],
            with_payload=True,
            with_vectors=False,
        )
        if res:
            return res[0].payload
    except Exception:
        pass
    return {}


@app.route("/", methods=["GET"])
def index():
    db = app.db_factory.db()
    wallet_tags = db.get_collections()
    wallets = [col.name for col in wallet_tags.collections]

    fingerprints = {}
    for wallet in wallets:
        fingerprints[wallet] = get_wallet_fingerprints(db, wallet)

    return render_template(
        "index.html",
        wallets=wallets,
        fingerprints=fingerprints,
        independent=HEURISTICS["independent"],
        probabilistic=HEURISTICS["probabilistic"],
        dependent=HEURISTICS["dependent"],
        temporal=HEURISTICS["temporal"],
    )


# Search route (commented out for now)
# @app.route("/submit", methods=["POST"])
# def submit():
#     user_input = request.form.get("user_input")
#     wallet = request.form.get("selected_wallet")
#     response = openai.embeddings.create(
#         input=user_input,
#         model="text-embedding-3-small"
#     )
#     user_input_embedding = np.array(
#         response.data[0].embedding, dtype=np.float32)
#
#     vector_db = app.db_factory.db()
#
#     res = vector_db.search(
#         collection_name=wallet,
#         query_vector=user_input_embedding,
#         limit=10
#     )
#
#     res = [{"function_str": result.payload["function_str"], "file_name": result.payload["file_name"]} for result in res]
#
#     return render_template("search-results.html", results=res)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wallet fingerprinting server.")
    parser.add_argument("--port", help="Port number", default=5000)
    parser.add_argument("--vector_db", help="Vector database uri",
                        default="http://localhost:6333")

    return parser.parse_args()


def main():
    args = parse_args()
    app.config["VECTOR_DB"] = args.vector_db
    db_factory = VectorDBFactory(app.config["VECTOR_DB"])
    app.db_factory = db_factory

    app.run(debug=True, port=args.port)


if __name__ == "__main__":
    main()
