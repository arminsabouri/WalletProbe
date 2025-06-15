from flask import Flask, render_template, request
import argparse
from qdrant_client import QdrantClient

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    user_input = request.form.get("user_input")
    return f"You submitted: {user_input}"

def parse_args():
    parser = argparse.ArgumentParser(description="Wallet fingerprinting server.")
    parser.add_argument("--port", help="Port number", default=5000)
    parser.add_argument("--vector_db", help="Vector database uri", default="http://localhost:6333")

    return parser.parse_args()

def main():
    args = parse_args()
    app.config["VECTOR_DB"] = args.vector_db
    qdrant_client = QdrantClient(url=app.config["VECTOR_DB"], prefer_grpc=False)
    app.qdrant_client = qdrant_client

    app.run(debug=True, port=args.port)

if __name__ == "__main__":
    main()
    