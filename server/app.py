from flask import Flask, render_template, request, jsonify
import argparse
from qdrant_client import QdrantClient
import numpy as np
import openai
import json

app = Flask(__name__)

class VectorDBFactory:
    def __init__(self, vector_db_uri):
        self.vector_db_uri = vector_db_uri

    def db(self):
        return QdrantClient(url=self.vector_db_uri, prefer_grpc=False)

@app.route("/", methods=["GET"])
def index():
    wallet_tags = app.db_factory.db().get_collections()
    wallets = [col.name for col in wallet_tags.collections]

    return render_template("index.html", wallets=wallets)


@app.route("/submit", methods=["POST"])
def submit():
    user_input = request.form.get("user_input")
    wallet = request.form.get("selected_wallet")
    response = openai.embeddings.create(
        input=user_input,
        model="text-embedding-3-small"
    )
    user_input_embedding = np.array(
        response.data[0].embedding, dtype=np.float32)

    vector_db = app.db_factory.db()

    res = vector_db.search(
        collection_name=wallet,
        query_vector=user_input_embedding,
        limit=10
    )

    print(res)

    res = [{"function_str": result.payload["function_str"], "file_name": result.payload["file_name"]} for result in res]

    return render_template("search-results.html", results=res)

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
