from flask import Flask, render_template, request
import argparse
from qdrant_client import QdrantClient

app = Flask(__name__)

class VectorDBFactory:
    def __init__(self, vector_db_uri):
        self.vector_db_uri = vector_db_uri

    def db(self):
        return QdrantClient(url=self.vector_db_uri, prefer_grpc=False)

@app.route("/", methods=["GET"])
def index():
    # wallets = wallets()
    wallet_tags = app.db_factory.db().get_collections()
    wallets = [col.name for col in wallet_tags.collections]


    print(wallets)
    return render_template("index.html", wallets=wallets)

@app.route("/submit", methods=["POST"])
def submit():
    user_input = request.form.get("user_input")
    return f"You submitted: {user_input}"

# Get list of wallets which we have embeddings for
# @app.route("/wallets", methods=["GET"])
def wallets():
    wallet_tags = app.db_factory.db().get_collections()
    collection_names = [col.name for col in wallet_tags.collections]

    return collection_names

def parse_args():
    parser = argparse.ArgumentParser(description="Wallet fingerprinting server.")
    parser.add_argument("--port", help="Port number", default=5000)
    parser.add_argument("--vector_db", help="Vector database uri", default="http://localhost:6333")

    return parser.parse_args()

def main():
    args = parse_args()
    app.config["VECTOR_DB"] = args.vector_db
    db_factory = VectorDBFactory(app.config["VECTOR_DB"])
    app.db_factory = db_factory

    app.run(debug=True, port=args.port)

if __name__ == "__main__":
    main()
    