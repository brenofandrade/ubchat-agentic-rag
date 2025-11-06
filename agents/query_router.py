"""
Query Router Agent

This agent receives a question and decides the best strategy to answer it:
- RAG: Query internal documents
- DIRECT: Answer with model's own knowledge
- CLARIFY: Ask follow-up questions to clarify the intent
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
import json
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)


class RouteType(str, Enum):
    """Types of routing decisions"""
    RAG = "rag"  # Query internal documents
    DIRECT = "direct"  # Answer with model knowledge
    CLARIFY = "clarify"  # Need more information


@dataclass
class RouteDecision:
    """Decision made by the query router"""
    route: RouteType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    clarifying_questions: Optional[List[str]] = None
    suggested_documents: Optional[List[str]] = None


class QueryRouter:
    """
    Intelligent query router that determines the best strategy to answer a question.

    Uses an LLM to analyze the question and decide whether to:
    - Query internal documents (RAG)
    - Answer directly with model knowledge (DIRECT)
    - Ask clarifying questions (CLARIFY)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", provider: str = "openai", base_url: Optional[str] = None):
        """
        Initialize the query router.

        Args:
            api_key: API key for the LLM provider (uses env var if not provided)
            model: Model to use (gpt-4, claude-3-opus-20240229, llama3.2:latest, etc.)
            provider: LLM provider ("openai", "anthropic", or "ollama")
            base_url: Base URL for Ollama (defaults to http://localhost:11434)
        """
        self.model = model
        self.provider = provider
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # API key only needed for OpenAI and Anthropic
        if provider in ["openai", "anthropic"]:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY")
        else:
            self.api_key = None

        # Import the appropriate client
        if provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key) if self.api_key else None
            except ImportError:
                self.client = None
        elif provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key) if self.api_key else None
            except ImportError:
                self.client = None
        elif provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
                self.client = ChatOllama(
                    model=self.model,
                    base_url=self.base_url,
                    temperature=0.1  # Baixa temperatura para respostas mais determinísticas
                )
            except ImportError:
                self.client = None
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'anthropic', or 'ollama'.")

    def route_query(self, question: str, context: Optional[str] = None) -> RouteDecision:
        """
        Analyze a question and decide the best routing strategy.

        Args:
            question: The user's question
            context: Optional context about the conversation or available documents

        Returns:
            RouteDecision with the routing choice and reasoning
        """
        logger.info(f"Roteando pergunta: '{question[:100]}...'")

        if not self.client:
            logger.warning("Cliente LLM não disponível, usando roteamento baseado em regras")
            # Fallback to rule-based routing if no LLM client available
            decision = self._rule_based_routing(question, context)
        else:
            # Use LLM for intelligent routing
            decision = self._llm_based_routing(question, context)

        logger.info(f"Decisão de roteamento: {decision.route.value} (confiança: {decision.confidence})")
        logger.debug(f"Raciocínio: {decision.reasoning}")

        return decision

    def _rule_based_routing(self, question: str, context: Optional[str] = None) -> RouteDecision:
        """
        Simple rule-based routing when LLM is not available.

        This is a fallback mechanism with improved heuristics.
        """
        logger.info("Usando roteamento baseado em regras (fallback)")
        question_lower = question.lower().strip()

        # Keywords que indicam fortemente busca em documentos internos (RAG)
        strong_rag_keywords = [
            "política", "policy", "políticas",
            "procedimento", "procedure", "procedimentos",
            "benefício", "benefit", "benefícios",
            "reembolso", "reimbursement",
            "férias", "vacation", "vacations",
            "home office", "trabalho remoto", "remote work",
            "rh", "hr", "recursos humanos", "human resources",
            "empresa", "company", "organização", "organization",
            "interno", "internal", "interna",
            "como solicito", "como solicitar", "how do i request",
            "qual o processo", "what is the process",
            "manual", "guide", "guia",
            "nossa", "nosso", "nossa empresa", "our company"
        ]

        # Keywords que indicam conhecimento geral (DIRECT)
        direct_keywords = [
            "o que é", "what is", "o que são", "what are",
            "como funciona", "how does", "how works",
            "defina", "define", "definição", "definition",
            "explique", "explain", "explicação", "explanation",
            "capital de", "capital of",
            "história", "history", "histórico",
            "ciência", "science", "científico",
            "matemática", "math", "mathematical",
            "física", "physics", "química", "chemistry",
            "biologia", "biology", "fotossíntese", "photosynthesis"
        ]

        # Perguntas extremamente vagas que requerem clarificação
        vague_patterns = [
            "como faço", "como fazer",
            "me ajuda", "help me", "ajuda",
            "preciso disso", "need this",
            "aquilo", "that thing", "isso aí",
            "aquele negócio"
        ]

        # 1. Perguntas muito curtas e vagas
        if len(question.strip()) < 8:
            if question_lower in ["oi", "olá", "hi", "hello", "help", "ajuda"]:
                return RouteDecision(
                    route=RouteType.CLARIFY,
                    confidence=0.9,
                    reasoning="Saudação ou pedido genérico de ajuda",
                    clarifying_questions=[
                        "Olá! Como posso ajudar você hoje?",
                        "Você tem alguma dúvida específica sobre a empresa ou algum tópico geral?"
                    ]
                )

        # 2. Check for vague patterns
        if any(vague in question_lower for vague in vague_patterns) and len(question.strip()) < 30:
            return RouteDecision(
                route=RouteType.CLARIFY,
                confidence=0.85,
                reasoning="Pergunta muito vaga sem contexto suficiente",
                clarifying_questions=[
                    "Pode ser mais específico sobre o que você precisa?",
                    "Você está perguntando sobre políticas da empresa ou sobre um tópico geral?"
                ]
            )

        # 3. Strong RAG indicators
        if any(keyword in question_lower for keyword in strong_rag_keywords):
            return RouteDecision(
                route=RouteType.RAG,
                confidence=0.85,
                reasoning="Pergunta sobre políticas/procedimentos da organização",
                suggested_documents=["company_policies", "procedures", "hr_manual"]
            )

        # 4. Direct knowledge indicators
        if any(keyword in question_lower for keyword in direct_keywords):
            return RouteDecision(
                route=RouteType.DIRECT,
                confidence=0.80,
                reasoning="Pergunta sobre conhecimento geral/conceitos"
            )

        # 5. Check for organizational context even without exact keywords
        org_context_words = ["empresa", "company", "trabalho", "work", "equipe", "team", "gestor", "manager"]
        if any(word in question_lower for word in org_context_words):
            return RouteDecision(
                route=RouteType.RAG,
                confidence=0.70,
                reasoning="Pergunta parece relacionada ao contexto organizacional",
                suggested_documents=["company_info", "policies"]
            )

        # 6. Default to DIRECT for general questions
        logger.info("Nenhuma regra específica aplicada, usando rota DIRECT como padrão")
        return RouteDecision(
            route=RouteType.DIRECT,
            confidence=0.60,
            reasoning="Pergunta parece ser de conhecimento geral (fallback padrão)"
        )

    def _llm_based_routing(self, question: str, context: Optional[str] = None) -> RouteDecision:
        """
        Use an LLM to intelligently route the query.
        """
        system_prompt = """Você é um agente de roteamento de consultas. Analise a pergunta do usuário e decida a MELHOR estratégia:

1. **RAG** - Use quando a pergunta precisa de DOCUMENTOS INTERNOS da organização:
   ✓ Políticas da empresa (férias, benefícios, RH, etc.)
   ✓ Procedimentos internos (reembolso, aprovações, processos)
   ✓ Informações específicas da organização
   ✓ Documentos, manuais, guias internos
   ✓ Palavras-chave: "empresa", "nossa política", "como solicito", "procedimento", "interno"

   Exemplos RAG:
   - "Qual é a política de férias da empresa?"
   - "Como solicito reembolso de despesas?"
   - "Quais são os benefícios oferecidos?"
   - "Qual o procedimento para home office?"

2. **DIRECT** - Use quando é CONHECIMENTO GERAL (não específico da organização):
   ✓ Conceitos científicos, históricos, matemáticos
   ✓ Definições gerais
   ✓ Conhecimento público/mundial
   ✓ Perguntas que qualquer pessoa poderia responder sem acessar documentos específicos

   Exemplos DIRECT:
   - "Como funciona fotossíntese?"
   - "Qual a capital da França?"
   - "O que é Python?"
   - "Explique o que é machine learning"

3. **CLARIFY** - Use APENAS quando a pergunta é IMPOSSÍVEL de entender:
   ✓ Pergunta extremamente vaga (ex: "Como faço?", "Me ajuda")
   ✓ Falta informação crítica para entender a intenção
   ✓ Múltiplas interpretações completamente diferentes

   Exemplos CLARIFY:
   - "Como faço?" (o quê?)
   - "Preciso disso" (disso o quê?)
   - "Aquilo ali" (qual coisa?)

IMPORTANTE: Prefira RAG ou DIRECT ao invés de CLARIFY. Use CLARIFY apenas em último caso.

Responda APENAS com JSON válido:
{
    "route": "rag" | "direct" | "clarify",
    "confidence": 0.0-1.0,
    "reasoning": "Breve explicação (1 frase)",
    "clarifying_questions": ["pergunta1", "pergunta2"] (opcional, apenas se route=clarify),
    "suggested_documents": ["tipo_doc1", "tipo_doc2"] (opcional, apenas se route=rag)
}"""

        user_prompt = f"Pergunta do usuário: {question}"
        if context:
            user_prompt += f"\n\nContexto adicional: {context}"

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=0.3,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                # Extract JSON from response
                content = response.content[0].text
                # Find JSON in the response
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    result = json.loads(content[json_start:json_end])
                else:
                    raise ValueError("No JSON found in response")

            elif self.provider == "ollama":
                # Ollama via LangChain
                full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nResposta (JSON):"
                logger.debug(f"Enviando prompt para Ollama: {full_prompt[:200]}...")
                response = self.client.invoke(full_prompt)

                # Extrai conteúdo da resposta
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)

                logger.debug(f"Resposta do Ollama: {content[:500]}...")

                # Parse JSON da resposta com tratamento de erros melhorado
                try:
                    # Tenta encontrar JSON na resposta
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1

                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        logger.debug(f"JSON extraído: {json_str}")
                        result = json.loads(json_str)
                    else:
                        # Se não encontrar JSON, tenta parsear diretamente
                        result = json.loads(content)

                    # Valida campos obrigatórios
                    if "route" not in result or "confidence" not in result or "reasoning" not in result:
                        raise ValueError(f"JSON inválido: faltam campos obrigatórios. Resposta: {content}")

                except json.JSONDecodeError as je:
                    logger.error(f"Erro ao parsear JSON do Ollama: {je}")
                    logger.error(f"Conteúdo recebido: {content}")
                    raise ValueError(f"Resposta do Ollama não é JSON válido: {content[:200]}")

            # Parse the result into a RouteDecision
            logger.info(f"Roteamento LLM bem-sucedido: {result.get('route', 'unknown')}")
            return RouteDecision(
                route=RouteType(result["route"]),
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                clarifying_questions=result.get("clarifying_questions"),
                suggested_documents=result.get("suggested_documents")
            )

        except Exception as e:
            logger.error(f"Erro durante roteamento LLM: {e}", exc_info=True)
            logger.warning("Caindo para roteamento baseado em regras")
            # Fallback to rule-based routing
            return self._rule_based_routing(question, context)

    def route_query_simple(self, question: str) -> str:
        """
        Simplified routing that just returns the route type as a string.

        Args:
            question: The user's question

        Returns:
            Route type as string: "rag", "direct", or "clarify"
        """
        decision = self.route_query(question)
        return decision.route.value
