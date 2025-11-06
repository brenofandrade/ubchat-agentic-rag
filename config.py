"""
Configuração centralizada do sistema UBChat Agentic RAG.
Todas as variáveis de ambiente são carregadas aqui.
"""

import os
import logging
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Configuração do logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Ollama Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "llama3.2:latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")

# --- Pinecone Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_DSUNIBLU")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX") or os.getenv("PINECONE_INDEX_NAME")
DEFAULT_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# --- Retrieval Configuration ---
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "2"))

# --- OpenAI (opcional, para fallback) ---
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# --- Reranking Configuration ---
RERANK_METHOD_DEFAULT = os.getenv("RERANK_METHOD_DEFAULT", "none").lower()
RERANK_TOP_K_DEFAULT = int(os.getenv("RERANK_TOP_K_DEFAULT", "0"))  # 0 significa usar top_k se não especificado
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "16"))

# --- Chat History Configuration ---
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
TTL_SETUP = int(os.getenv("TTL_SETUP", "1200"))

# --- Validações ---
def validate_config():
    """Valida configurações críticas."""
    errors = []

    if not PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY não configurada.")

    if not PINECONE_INDEX_NAME:
        errors.append("PINECONE_INDEX (ou PINECONE_INDEX_NAME) não configurada.")

    if errors:
        error_msg = "\n".join(errors)
        raise RuntimeError(f"Erro de configuração:\n{error_msg}")

    logger.info("✓ Configurações validadas com sucesso")
    logger.info(f"  - Ollama URL: {OLLAMA_BASE_URL}")
    logger.info(f"  - Modelo de Geração: {GENERATION_MODEL}")
    logger.info(f"  - Modelo de Embedding: {EMBEDDING_MODEL}")
    logger.info(f"  - Pinecone Index: {PINECONE_INDEX_NAME}")
    logger.info(f"  - Namespace: {DEFAULT_NAMESPACE}")
    logger.info(f"  - Retrieval K: {RETRIEVAL_K}")
    logger.info(f"  - Rerank Method: {RERANK_METHOD_DEFAULT}")

# Executa validação ao importar
if __name__ != "__main__":
    validate_config()
