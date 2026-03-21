from flask import Flask, render_template
import argparse
from heuristics import HEURISTICS
from qdrant_utils import VectorDBFactory, get_wallet_fingerprints
# import numpy as np
# import openai

app = Flask(__name__)


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
