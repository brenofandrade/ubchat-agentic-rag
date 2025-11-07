# Agentic RAG using LangGraph served by API Flask

import os
import re
import json
import time
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import configuration
from config import (
    GENERATION_MODEL,
    OLLAMA_BASE_URL,
    PINECONE_INDEX_NAME,
    DEFAULT_NAMESPACE,
    LOG_LEVEL
)

# Import the query router agent and RAG engine
from agents import QueryRouter, RouteDecision, RouteType, get_rag_engine

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



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

# Initialize the query router agent with Ollama
# You can configure the provider and model via environment variables
logger.info("Inicializando Query Router e RAG Engine...")

query_router = QueryRouter(
    model=os.getenv("GENERATION_MODEL", GENERATION_MODEL),
    provider=os.getenv("LLM_PROVIDER", "ollama"),
    base_url=OLLAMA_BASE_URL
)

# Initialize RAG engine (singleton)
rag_engine = get_rag_engine(namespace=DEFAULT_NAMESPACE)

logger.info(f"✓ Sistema inicializado com sucesso")
logger.info(f"  - Provider: {query_router.provider}")
logger.info(f"  - Model: {query_router.model}")
logger.info(f"  - Pinecone Index: {PINECONE_INDEX_NAME}")
logger.info(f"  - Namespace: {DEFAULT_NAMESPACE}")

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    print("[ROTA] GET /health - Health check acionado")
    return jsonify({
        "status": "ok",
        "provider": query_router.provider,
        "model": query_router.model,
        "pinecone_index": PINECONE_INDEX_NAME,
        "namespace": DEFAULT_NAMESPACE
    }), 200


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
    print("[ROTA] POST /route-query - Roteamento de query acionado")
    try:
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({
                "error": "Missing 'question' field in request body"
            }), 400

        question = data["question"]
        print(f"  └─ Pergunta: {question}")
        context = data.get("context")

        # Route the query
        decision = query_router.route_query(question, context)
        print(f"  └─ Decisão de roteamento: {decision.route.value} (confiança: {decision.confidence:.2f})")

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
    print("[ROTA] POST /route-query/simple - Roteamento simples acionado")
    try:
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({
                "error": "Missing 'question' field in request body"
            }), 400

        question = data["question"]
        print(f"  └─ Pergunta: {question}")
        route = query_router.route_query_simple(question)
        print(f"  └─ Rota decidida: {route}")

        return jsonify({"route": route}), 200

    except Exception as e:
        return jsonify({
            "error": f"Error processing request: {str(e)}"
        }), 500



@app.route("/rag/query", methods=["POST"])
def rag_query():
    """
    Execute a complete RAG query: retrieval + generation.

    Expected JSON body:
    {
        "question": "User's question",
        "chat_history": [{"role": "user", "content": "..."}, ...] (optional),
        "top_k": 5 (optional, number of documents to retrieve),
        "namespace": "custom_namespace" (optional)
    }

    Returns:
    {
        "answer": "Generated answer",
        "documents": [
            {
                "content": "Document content",
                "metadata": {...},
                "score": 0.95
            },
            ...
        ],
        "metadata": {
            "retrieved_count": 5,
            "generation_model": "llama3.2:latest",
            "embedding_model": "mxbai-embed-large:latest",
            "namespace": "default"
        }
    }
    """
    print("[ROTA] POST /rag/query - Query RAG completa acionada")
    try:
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({
                "error": "Missing 'question' field in request body"
            }), 400

        question = data["question"]
        print(f"  └─ Pergunta: {question}")
        chat_history = data.get("chat_history", [])
        top_k = data.get("top_k")
        namespace = data.get("namespace", DEFAULT_NAMESPACE)

        # Get RAG engine for the specified namespace
        if namespace != DEFAULT_NAMESPACE:
            engine = get_rag_engine(namespace=namespace, force_new=True)
        else:
            engine = rag_engine

        # Execute RAG query
        result = engine.query(
            query=question,
            chat_history=chat_history,
            top_k=top_k
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Erro ao processar query RAG: {e}", exc_info=True)
        return jsonify({
            "error": f"Error processing RAG query: {str(e)}"
        }), 500


@app.route("/rag/retrieve", methods=["POST"])
def rag_retrieve():
    """
    Retrieve relevant documents without generating an answer.

    Expected JSON body:
    {
        "question": "User's question",
        "top_k": 5 (optional),
        "namespace": "custom_namespace" (optional)
    }

    Returns:
    {
        "documents": [
            {
                "content": "Document content",
                "metadata": {...},
                "score": 0.95
            },
            ...
        ],
        "count": 5
    }
    """
    print("[ROTA] POST /rag/retrieve - Recuperação de documentos acionada")
    try:
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({
                "error": "Missing 'question' field in request body"
            }), 400

        question = data["question"]
        print(f"  └─ Pergunta: {question}")
        top_k = data.get("top_k")
        namespace = data.get("namespace", DEFAULT_NAMESPACE)

        # Get RAG engine for the specified namespace
        if namespace != DEFAULT_NAMESPACE:
            engine = get_rag_engine(namespace=namespace, force_new=True)
        else:
            engine = rag_engine

        # Retrieve documents
        documents = engine.retrieve(query=question, top_k=top_k)

        # Convert to JSON-serializable format
        result = {
            "documents": [
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": doc.score
                }
                for doc in documents
            ],
            "count": len(documents)
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Erro ao recuperar documentos: {e}", exc_info=True)
        return jsonify({
            "error": f"Error retrieving documents: {str(e)}"
        }), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Complete chat endpoint that uses query routing + RAG.

    Expected JSON body:
    {
        "question": "User's question",
        "chat_history": [{"role": "user", "content": "..."}, ...] (optional),
        "context": "Optional context" (optional)
    }

    Returns:
    {
        "answer": "Generated answer or clarifying questions",
        "route": "rag" | "direct" | "clarify",
        "confidence": 0.95,
        "reasoning": "Explanation",
        "documents": [...] (if route=rag),
        "clarifying_questions": [...] (if route=clarify)
    }
    """
    print("[ROTA] POST /chat - Chat completo acionado")
    try:
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({
                "error": "Missing 'question' field in request body"
            }), 400

        question = data["question"]
        print(f"  └─ Pergunta: {question}")
        chat_history = data.get("chat_history", [])
        context = data.get("context")

        # Route the query
        decision = query_router.route_query(question, context)
        print(f"  └─ Decisão de roteamento: {decision.route.value} (confiança: {decision.confidence:.2f})")

        # Build response based on route
        response = {
            "route": decision.route.value,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning
        }

        # Handle different routes
        if decision.route == RouteType.RAG:
            # Execute RAG query
            rag_result = rag_engine.query(
                query=question,
                chat_history=chat_history
            )
            response["answer"] = rag_result["answer"]
            response["documents"] = rag_result["documents"]

        elif decision.route == RouteType.DIRECT:
            # Use LLM directly (without RAG)
            # For now, we'll use the RAG engine's LLM but without documents
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model=GENERATION_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.3
            )

            prompt = f"Responda a seguinte pergunta com base em seu conhecimento:\n\nPergunta: {question}\n\nResposta:"
            llm_response = llm.invoke(prompt)

            if hasattr(llm_response, 'content'):
                answer = llm_response.content
            else:
                answer = str(llm_response)

            response["answer"] = answer

        elif decision.route == RouteType.CLARIFY:
            # Return clarifying questions
            response["clarifying_questions"] = decision.clarifying_questions
            response["answer"] = "Preciso de mais informações para responder adequadamente."

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Erro ao processar chat: {e}", exc_info=True)
        return jsonify({
            "error": f"Error processing chat: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8000", debug=True)