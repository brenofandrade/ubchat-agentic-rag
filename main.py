# Agentic RAG using LangGraph served by API Flask

import os
import re
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS



app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
    max_age=86400,
)
app.url_map.strict_slashes = False

@app.route("/health", methods=["GET"])
def health():
    
    return jsonify({"status": "ok"}), 200






if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8000", debug=True)