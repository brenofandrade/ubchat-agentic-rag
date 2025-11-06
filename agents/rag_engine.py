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

    def retrieve(self, query: str, top_k: Optional[int] = None, enable_fallback: bool = True) -> List[RetrievedDocument]:
        """
        Recupera documentos relevantes do vector store.

        Args:
            query: Consulta do usuário
            top_k: Número de documentos a recuperar (sobrescreve padrão)
            enable_fallback: Se True, tenta buscar com termos alternativos quando não encontrar resultados

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

            # Se não encontrou documentos e fallback está habilitado, tenta com variações
            if not documents and enable_fallback:
                logger.warning("Nenhum documento encontrado. Tentando com variações da consulta...")
                documents = self._retrieve_with_variations(query, k)

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

    def _generate_query_variations(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Gera variações da consulta original para expandir a busca.

        Args:
            query: Consulta original
            num_variations: Número de variações a gerar

        Returns:
            Lista de variações da consulta
        """
        logger.info(f"Gerando {num_variations} variações para: '{query[:100]}...'")

        # Tenta usar OpenAI se disponível
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "Você é um assistente que gera variações de consultas de busca em português. "
                                       "Para cada consulta fornecida, gere variações usando sinônimos, "
                                       "termos relacionados e diferentes formas de expressar a mesma ideia."
                        },
                        {
                            "role": "user",
                            "content": f"Gere {num_variations} variações diferentes da seguinte consulta de busca:\n\n"
                                       f"Consulta: {query}\n\n"
                                       f"Retorne apenas as variações, uma por linha, sem numeração."
                        }
                    ],
                    temperature=0.7,
                    max_tokens=200
                )

                variations_text = response.choices[0].message.content.strip()
                variations = [v.strip() for v in variations_text.split('\n') if v.strip()]
                logger.info(f"✓ Geradas {len(variations)} variações via OpenAI")
                return variations[:num_variations]

            except Exception as e:
                logger.warning(f"Erro ao gerar variações com OpenAI: {e}, usando fallback")

        # Fallback: usa Ollama
        try:
            prompt = f"""Gere {num_variations} variações diferentes da seguinte consulta de busca em português.
Use sinônimos, termos relacionados e diferentes formas de expressar a mesma ideia.

Consulta original: {query}

Retorne apenas as variações, uma por linha, sem numeração ou explicações."""

            response = self.llm.invoke(prompt)

            if hasattr(response, 'content'):
                variations_text = response.content
            else:
                variations_text = str(response)

            variations = [v.strip() for v in variations_text.split('\n') if v.strip()]
            logger.info(f"✓ Geradas {len(variations)} variações via Ollama")
            return variations[:num_variations]

        except Exception as e:
            logger.error(f"Erro ao gerar variações com Ollama: {e}")

            # Fallback simples: variações baseadas em regras
            logger.info("Usando fallback baseado em regras para variações")
            return self._generate_simple_variations(query)

    def _generate_simple_variations(self, query: str) -> List[str]:
        """
        Gera variações simples da consulta usando regras básicas.

        Args:
            query: Consulta original

        Returns:
            Lista de variações simples
        """
        variations = []
        query_lower = query.lower()

        # Dicionário de sinônimos comuns em português
        synonyms = {
            "como": ["de que forma", "de que maneira"],
            "política": ["regra", "norma", "procedimento"],
            "solicitar": ["pedir", "requisitar", "fazer pedido"],
            "benefício": ["vantagem", "privilégio"],
            "férias": ["período de descanso", "recesso"],
            "reembolso": ["ressarcimento", "devolução"],
            "trabalho remoto": ["home office", "trabalho em casa"],
            "empresa": ["organização", "companhia"],
            "processo": ["procedimento", "método"]
        }

        # Tenta substituir palavras por sinônimos
        for word, syns in synonyms.items():
            if word in query_lower:
                for syn in syns[:2]:  # Limita a 2 sinônimos
                    variation = query_lower.replace(word, syn)
                    if variation != query_lower:
                        variations.append(variation)

        # Se não gerou variações, retorna consulta simplificada
        if not variations:
            # Remove palavras de parada
            stop_words = ["como", "qual", "quais", "o", "a", "os", "as", "de", "da", "do"]
            words = query_lower.split()
            filtered = [w for w in words if w not in stop_words]
            if filtered:
                variations.append(" ".join(filtered))

        logger.info(f"✓ Geradas {len(variations)} variações simples")
        return variations[:3]

    def _retrieve_with_variations(self, query: str, k: int) -> List[RetrievedDocument]:
        """
        Tenta recuperar documentos usando variações da consulta original.

        Args:
            query: Consulta original
            k: Número de documentos a recuperar

        Returns:
            Lista de documentos recuperados
        """
        variations = self._generate_query_variations(query, num_variations=3)

        all_documents = []
        seen_content = set()

        for i, variation in enumerate(variations, 1):
            logger.info(f"Tentando variação {i}/{len(variations)}: '{variation[:100]}...'")

            try:
                if self.use_hybrid and self.sparse_encoder:
                    docs = self._hybrid_retrieve(variation, k)
                    if not docs:
                        docs = self._dense_retrieve(variation, k)
                else:
                    docs = self._dense_retrieve(variation, k)

                # Adiciona apenas documentos únicos
                for doc in docs:
                    # Usa hash do conteúdo para verificar duplicatas
                    content_hash = hash(doc.content)
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_documents.append(doc)

                # Se encontrou documentos suficientes, para
                if len(all_documents) >= k:
                    logger.info(f"✓ Encontrados {len(all_documents)} documentos com variações")
                    break

            except Exception as e:
                logger.error(f"Erro ao buscar com variação '{variation}': {e}")
                continue

        # Ordena por score e retorna top k
        all_documents.sort(key=lambda x: x.score, reverse=True)
        return all_documents[:k]

    def _generate_search_suggestions(self, query: str) -> List[str]:
        """
        Gera sugestões de termos alternativos para ajudar o usuário quando não encontrar resultados.

        Args:
            query: Consulta original que não retornou resultados

        Returns:
            Lista de sugestões de busca
        """
        logger.info(f"Gerando sugestões de busca para: '{query[:100]}...'")

        # Tenta usar OpenAI se disponível
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "Você é um assistente que ajuda usuários a refinar buscas. "
                                       "Quando uma busca não retorna resultados, sugira 3 termos ou frases "
                                       "alternativos que o usuário poderia tentar. Seja específico e prático."
                        },
                        {
                            "role": "user",
                            "content": f"A seguinte busca não retornou resultados:\n\n'{query}'\n\n"
                                       f"Sugira 3 termos ou frases alternativos que o usuário poderia buscar. "
                                       f"Retorne apenas as sugestões, uma por linha."
                        }
                    ],
                    temperature=0.7,
                    max_tokens=150
                )

                suggestions_text = response.choices[0].message.content.strip()
                suggestions = [s.strip().strip('-•').strip() for s in suggestions_text.split('\n') if s.strip()]
                logger.info(f"✓ Geradas {len(suggestions)} sugestões via OpenAI")
                return suggestions[:3]

            except Exception as e:
                logger.warning(f"Erro ao gerar sugestões com OpenAI: {e}, usando fallback")

        # Fallback: usa Ollama
        try:
            prompt = f"""A seguinte busca não retornou resultados: "{query}"

Sugira 3 termos ou frases alternativos que o usuário poderia buscar para encontrar informações relacionadas.
Seja específico e prático. Retorne apenas as sugestões, uma por linha, sem numeração."""

            response = self.llm.invoke(prompt)

            if hasattr(response, 'content'):
                suggestions_text = response.content
            else:
                suggestions_text = str(response)

            suggestions = [s.strip().strip('-•').strip() for s in suggestions_text.split('\n') if s.strip()]
            logger.info(f"✓ Geradas {len(suggestions)} sugestões via Ollama")
            return suggestions[:3]

        except Exception as e:
            logger.error(f"Erro ao gerar sugestões com Ollama: {e}")

            # Fallback simples: usa variações geradas anteriormente
            logger.info("Usando variações como sugestões (fallback)")
            return self._generate_simple_variations(query)

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
            # Gera sugestões de termos alternativos para ajudar o usuário
            suggestions = self._generate_search_suggestions(query)

            answer_parts = [
                "Desculpe, não encontrei documentos relevantes para sua pergunta.",
                ""
            ]

            if suggestions:
                answer_parts.append("Você poderia tentar buscar por:")
                for i, suggestion in enumerate(suggestions, 1):
                    answer_parts.append(f"{i}. {suggestion}")
                answer_parts.append("")

            answer_parts.extend([
                "Ou você pode fornecer mais detalhes sobre:",
                "- O contexto da sua pergunta",
                "- Termos específicos relacionados ao que você procura",
                "- Uma reformulação da sua pergunta com mais informações"
            ])

            return {
                "answer": "\n".join(answer_parts),
                "documents": [],
                "metadata": {
                    "retrieved_count": 0,
                    "generation_model": GENERATION_MODEL,
                    "embedding_model": EMBEDDING_MODEL,
                    "search_suggestions": suggestions
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
