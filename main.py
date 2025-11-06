# Agentic RAG using LangGraph served by API Flask

import os
import re
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import the query router agent
from agents import QueryRouter, RouteDecision, RouteType



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

# Initialize the query router agent
# You can configure the provider and model via environment variables
query_router = QueryRouter(
    model=os.getenv("LLM_MODEL", "gpt-4"),
    provider=os.getenv("LLM_PROVIDER", "openai")
)

@app.route("/health", methods=["GET"])
def health():

    return jsonify({"status": "ok"}), 200


@app.route("/route-query", methods=["POST"])
def route_query():
    """
    Route a query to determine the best strategy to answer it.

    Expected JSON body:
    {
        "question": "User's question",
        "context": "Optional context" (optional)
    }

    Returns:
    {
        "route": "rag" | "direct" | "clarify",
        "confidence": 0.0-1.0,
        "reasoning": "Explanation",
        "clarifying_questions": ["question1", "question2"] (if route=clarify),
        "suggested_documents": ["doc1", "doc2"] (if route=rag)
    }
    """
    try:
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({
                "error": "Missing 'question' field in request body"
            }), 400

        question = data["question"]
        context = data.get("context")

        # Route the query
        decision = query_router.route_query(question, context)

        # Convert to JSON-serializable format
        response = {
            "route": decision.route.value,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning
        }

        if decision.clarifying_questions:
            response["clarifying_questions"] = decision.clarifying_questions

        if decision.suggested_documents:
            response["suggested_documents"] = decision.suggested_documents

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "error": f"Error processing request: {str(e)}"
        }), 500


@app.route("/route-query/simple", methods=["POST"])
def route_query_simple():
    """
    Simplified route endpoint that just returns the route type.

    Expected JSON body:
    {
        "question": "User's question"
    }

    Returns:
    {
        "route": "rag" | "direct" | "clarify"
    }
    """
    try:
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({
                "error": "Missing 'question' field in request body"
            }), 400

        question = data["question"]
        route = query_router.route_query_simple(question)

        return jsonify({"route": route}), 200

    except Exception as e:
        return jsonify({
            "error": f"Error processing request: {str(e)}"
        }), 500






if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8000", debug=True)