"""
Motor RAG (Retrieval-Augmented Generation) usando Ollama e Pinecone.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from pinecone import Pinecone
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI

try:
    from pinecone_text.hybrid import hybrid_convex_scale
    from pinecone_text.sparse import SpladeSparseEncoder
except ImportError:  # pragma: no cover - fallback when library is missing
    hybrid_convex_scale = None
    SpladeSparseEncoder = None

# Importa configurações
from config import (
    OLLAMA_BASE_URL,
    GENERATION_MODEL,
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    DEFAULT_NAMESPACE,
    RETRIEVAL_K,
    OPENAI_KEY,
    RERANK_METHOD_DEFAULT,
    RERANK_TOP_K_DEFAULT,
    CROSS_ENCODER_MODEL,
    RERANK_BATCH_SIZE,
    MAX_HISTORY,
    ENABLE_HYBRID_SEARCH,
    HYBRID_ALPHA,
    SPLADE_MODEL
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """Documento recuperado do vector store."""
    content: str
    metadata: Dict[str, Any]
    score: float

    def __repr__(self):
        return f"RetrievedDocument(score={self.score:.3f}, content='{self.content[:50]}...')"


class RAGEngine:
    """
    Motor de Retrieval-Augmented Generation usando Ollama e Pinecone.

    Características:
    - Embeddings locais via Ollama
    - Vector store em nuvem via Pinecone
    - Geração de resposta via Ollama
    - Suporte a reranking (opcional)
    - Suporte a variações de consulta via OpenAI (opcional)
    """

    def __init__(
        self,
        namespace: str = DEFAULT_NAMESPACE,
        top_k: int = RETRIEVAL_K,
        rerank_method: str = RERANK_METHOD_DEFAULT,
        rerank_top_k: int = RERANK_TOP_K_DEFAULT
    ):
        """
        Inicializa o motor RAG.

        Args:
            namespace: Namespace do Pinecone para isolar documentos
            top_k: Número de documentos a recuperar
            rerank_method: Método de reranking ('none', 'cross-encoder')
            rerank_top_k: Número de documentos após reranking (0 = usar top_k)
        """
        self.namespace = namespace
        self.top_k = top_k
        self.rerank_method = rerank_method
        self.rerank_top_k = rerank_top_k if rerank_top_k > 0 else top_k
        self.use_hybrid = ENABLE_HYBRID_SEARCH
        self.hybrid_alpha = HYBRID_ALPHA
        self.sparse_encoder = None

        logger.info(f"Inicializando RAG Engine com namespace '{namespace}'")

        if self.use_hybrid:
            if SpladeSparseEncoder is None or hybrid_convex_scale is None:
                logger.warning(
                    "Biblioteca 'pinecone-text' não disponível; desativando busca híbrida."
                )
                self.use_hybrid = False
            else:
                try:
                    logger.info(
                        "Carregando codificador esparso para busca híbrida: %s",
                        SPLADE_MODEL
                    )
                    self.sparse_encoder = SpladeSparseEncoder(model_name=SPLADE_MODEL)
                    logger.info(
                        "Busca híbrida ativada (alpha=%.2f)",
                        self.hybrid_alpha
                    )
                except Exception as exc:
                    logger.error(f"Erro ao carregar encoder esparso: {exc}")
                    self.use_hybrid = False

        # Inicializa cliente OpenAI (opcional, para variações de consulta)
        self.openai_client = None
        if OPENAI_KEY:
            try:
                self.openai_client = OpenAI(api_key=OPENAI_KEY)
                logger.info("Cliente OpenAI inicializado (para variações de consulta)")
            except Exception as e:
                logger.warning(f"Falha ao inicializar OpenAI: {e}")
        else:
            logger.info("OPENAI_API_KEY não configurada; usando fallback para variações")

        # Inicializa Pinecone
        logger.info(f"Conectando ao Pinecone index '{PINECONE_INDEX_NAME}'")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        self._verify_pinecone_connection()

        # Inicializa LLM Ollama para geração
        logger.info(f"Inicializando Ollama LLM: {GENERATION_MODEL}")
        self.llm = ChatOllama(
            model=GENERATION_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0
        )

        # Inicializa embeddings Ollama
        logger.info(f"Inicializando Ollama Embeddings: {EMBEDDING_MODEL}")
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )

        # Cria a conexão com o Vectorstore
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            namespace=self.namespace
        )

        # Inicializa reranker se necessário
        self.reranker = None
        if self.rerank_method == "cross-encoder":
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Carregando Cross-Encoder: {CROSS_ENCODER_MODEL}")
                self.reranker = CrossEncoder(CROSS_ENCODER_MODEL)
            except Exception as e:
                logger.error(f"Erro ao carregar Cross-Encoder: {e}")
                self.rerank_method = "none"

        logger.info("✓ RAG Engine inicializado com sucesso")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDocument]:
        """
        Recupera documentos relevantes do vector store.

        Args:
            query: Consulta do usuário
            top_k: Número de documentos a recuperar (sobrescreve padrão)

        Returns:
            Lista de documentos recuperados
        """
        k = top_k if top_k is not None else self.top_k

        logger.info(f"Recuperando {k} documentos para query: '{query[:100]}...'")

        try:
            if self.use_hybrid and self.sparse_encoder:
                documents = self._hybrid_retrieve(query, k)
                if not documents:
                    logger.warning(
                        "Busca híbrida não retornou resultados; usando busca densa como fallback."
                    )
                    documents = self._dense_retrieve(query, k)
            else:
                documents = self._dense_retrieve(query, k)

            # Aplica reranking se configurado
            if self.rerank_method != "none" and len(documents) > 0:
                documents = self._rerank(query, documents)

            return documents

        except Exception as e:
            logger.error(f"Erro ao recuperar documentos: {e}")
            return []

    def _dense_retrieve(self, query: str, k: int) -> List[RetrievedDocument]:
        """Executa recuperação baseada apenas em embeddings densos."""
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            namespace=self.namespace
        )

        documents = [
            RetrievedDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score
            )
            for doc, score in results
        ]

        logger.info(f"✓ Recuperados {len(documents)} documentos (busca densa)")
        return documents

    def _hybrid_retrieve(self, query: str, k: int) -> List[RetrievedDocument]:
        """Executa recuperação híbrida combinando sinais densos e esparsos."""
        try:
            dense_vector = self.embeddings.embed_query(query)
        except Exception as exc:
            logger.error(f"Erro ao gerar embedding para busca híbrida: {exc}")
            return []

        try:
            sparse_vector = self.sparse_encoder.encode_queries([query])[0]
        except Exception as exc:
            logger.error(f"Erro ao gerar vetor esparso para busca híbrida: {exc}")
            return []

        if hybrid_convex_scale:
            try:
                dense_vector, sparse_vector = hybrid_convex_scale(
                    dense_vector,
                    sparse_vector,
                    self.hybrid_alpha
                )
            except Exception as exc:
                logger.error(f"Erro ao aplicar escala híbrida: {exc}")
                return []

        try:
            response = self.index.query(
                vector=dense_vector,
                sparse_vector=sparse_vector,
                namespace=self.namespace,
                top_k=k,
                include_metadata=True
            )
        except Exception as exc:
            logger.error(f"Erro ao consultar Pinecone (busca híbrida): {exc}")
            return []

        matches = getattr(response, "matches", []) or []
        documents: List[RetrievedDocument] = []

        for match in matches:
            metadata = getattr(match, "metadata", None)
            if metadata is None and isinstance(match, dict):
                metadata = match.get("metadata", {})
            metadata = metadata or {}

            content = (
                metadata.get("text")
                or metadata.get("page_content")
                or metadata.get("content")
                or metadata.get("chunk")
                or metadata.get("body")
                or ""
            )

            if not content and isinstance(match, dict):
                content = match.get("text", "")

            score = getattr(match, "score", None)
            if score is None and isinstance(match, dict):
                score = match.get("score", 0.0)

            try:
                score_value = float(score) if score is not None else 0.0
            except (TypeError, ValueError):
                score_value = 0.0

            documents.append(
                RetrievedDocument(
                    content=content,
                    metadata=metadata,
                    score=score_value
                )
            )

        logger.info(f"✓ Recuperados {len(documents)} documentos (busca híbrida)")
        return documents

    def _verify_pinecone_connection(self) -> None:
        """Verifica se a conexão com o Pinecone está operacional."""
        try:
            stats = self.index.describe_index_stats(namespace=self.namespace)
            dimension = None
            if isinstance(stats, dict):
                dimension = stats.get("dimension")
            else:
                dimension = getattr(stats, "dimension", None)
            logger.info(
                "✓ Conexão com Pinecone verificada (dimensão: %s)",
                dimension if dimension is not None else "desconhecida"
            )
        except Exception as exc:
            logger.error(f"Falha ao verificar conexão com Pinecone: {exc}")
            raise RuntimeError("Não foi possível verificar a conexão com o Pinecone.") from exc

    def _rerank(self, query: str, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        Reordena documentos usando cross-encoder.

        Args:
            query: Consulta original
            documents: Documentos a reordenar

        Returns:
            Documentos reordenados
        """
        if not self.reranker:
            logger.warning("Reranker não disponível, pulando reranking")
            return documents

        logger.info(f"Aplicando reranking com {self.rerank_method}")

        try:
            # Prepara pares (query, documento)
            pairs = [(query, doc.content) for doc in documents]

            # Calcula scores de relevância
            scores = self.reranker.predict(pairs, batch_size=RERANK_BATCH_SIZE)

            # Atualiza scores e reordena
            for doc, score in zip(documents, scores):
                doc.score = float(score)

            # Ordena por score (maior primeiro) e pega top_k
            reranked = sorted(documents, key=lambda x: x.score, reverse=True)
            reranked = reranked[:self.rerank_top_k]

            logger.info(f"✓ Reranking concluído: {len(reranked)} documentos")
            return reranked

        except Exception as e:
            logger.error(f"Erro no reranking: {e}")
            return documents

    def generate_answer(
        self,
        query: str,
        documents: List[RetrievedDocument],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Gera resposta usando documentos recuperados.

        Args:
            query: Pergunta do usuário
            documents: Documentos relevantes
            chat_history: Histórico de conversa (opcional)

        Returns:
            Resposta gerada
        """
        logger.info(f"Gerando resposta com {len(documents)} documentos")

        # Prepara contexto dos documentos
        context = "\n\n".join([
            f"Documento {i+1} (score: {doc.score:.3f}):\n{doc.content}"
            for i, doc in enumerate(documents)
        ])

        # Prepara histórico de conversa
        history_text = ""
        if chat_history:
            history_items = chat_history[-MAX_HISTORY:]  # Limita histórico
            history_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in history_items
            ])
            history_text = f"\n\nHistórico da conversa:\n{history_text}\n"

        # Monta prompt
        prompt = f"""Você é um assistente prestativo que responde perguntas com base em documentos fornecidos.

{history_text}
Contexto dos documentos:
{context}

Pergunta do usuário: {query}

Instruções:
1. Responda APENAS com base nos documentos fornecidos
2. Se os documentos não contiverem informação suficiente, diga claramente
3. Cite os documentos quando relevante (ex: "De acordo com o Documento 1...")
4. Seja objetivo e direto
5. Responda em português brasileiro

Resposta:"""

        try:
            # Gera resposta via Ollama
            response = self.llm.invoke(prompt)

            # Extrai texto da resposta
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)

            logger.info("✓ Resposta gerada com sucesso")
            return answer

        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {e}")
            return f"Desculpe, ocorreu um erro ao gerar a resposta: {str(e)}"

    def query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Pipeline completo RAG: recuperação + geração.

        Args:
            query: Pergunta do usuário
            chat_history: Histórico de conversa
            top_k: Número de documentos a recuperar

        Returns:
            Dicionário com resposta e metadados
        """
        logger.info(f"Executando query RAG: '{query[:100]}...'")

        # Recupera documentos
        documents = self.retrieve(query, top_k=top_k)

        if not documents:
            return {
                "answer": "Desculpe, não encontrei documentos relevantes para sua pergunta.",
                "documents": [],
                "metadata": {
                    "retrieved_count": 0,
                    "generation_model": GENERATION_MODEL,
                    "embedding_model": EMBEDDING_MODEL
                }
            }

        # Gera resposta
        answer = self.generate_answer(query, documents, chat_history)

        # Retorna resultado completo
        return {
            "answer": answer,
            "documents": [
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": doc.score
                }
                for doc in documents
            ],
            "metadata": {
                "retrieved_count": len(documents),
                "generation_model": GENERATION_MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "namespace": self.namespace,
                "rerank_method": self.rerank_method
            }
        }


# Instância global (singleton)
_rag_engine_instance = None


def get_rag_engine(
    namespace: str = DEFAULT_NAMESPACE,
    force_new: bool = False
) -> RAGEngine:
    """
    Retorna instância singleton do RAG Engine.

    Args:
        namespace: Namespace do Pinecone
        force_new: Força criação de nova instância

    Returns:
        Instância do RAG Engine
    """
    global _rag_engine_instance

    if _rag_engine_instance is None or force_new:
        _rag_engine_instance = RAGEngine(namespace=namespace)

    return _rag_engine_instance
