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

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", provider: str = "openai"):
        """
        Initialize the query router.

        Args:
            api_key: API key for the LLM provider (uses env var if not provided)
            model: Model to use (gpt-4, claude-3-opus-20240229, etc.)
            provider: LLM provider ("openai" or "anthropic")
        """
        self.model = model
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY")

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
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def route_query(self, question: str, context: Optional[str] = None) -> RouteDecision:
        """
        Analyze a question and decide the best routing strategy.

        Args:
            question: The user's question
            context: Optional context about the conversation or available documents

        Returns:
            RouteDecision with the routing choice and reasoning
        """
        if not self.client:
            # Fallback to rule-based routing if no LLM client available
            return self._rule_based_routing(question, context)

        # Use LLM for intelligent routing
        return self._llm_based_routing(question, context)

    def _rule_based_routing(self, question: str, context: Optional[str] = None) -> RouteDecision:
        """
        Simple rule-based routing when LLM is not available.

        This is a fallback mechanism with basic heuristics.
        """
        question_lower = question.lower()

        # Keywords that suggest internal document lookup
        rag_keywords = [
            "documento", "document", "arquivo", "file", "política", "policy",
            "procedimento", "procedure", "manual", "guia", "guide",
            "nossa empresa", "our company", "nosso", "our", "interno", "internal"
        ]

        # Keywords that suggest clarification needed
        clarify_keywords = [
            "?", "qual", "what", "como", "how", "quando", "when",
            "onde", "where", "quem", "who", "por que", "why"
        ]

        # Very short or vague questions
        if len(question.strip()) < 10:
            return RouteDecision(
                route=RouteType.CLARIFY,
                confidence=0.8,
                reasoning="Pergunta muito curta, precisa de mais detalhes",
                clarifying_questions=[
                    "Você pode fornecer mais detalhes sobre o que precisa?",
                    "Em que contexto você tem essa dúvida?"
                ]
            )

        # Check for RAG keywords
        if any(keyword in question_lower for keyword in rag_keywords):
            return RouteDecision(
                route=RouteType.RAG,
                confidence=0.7,
                reasoning="Pergunta parece relacionada a documentos internos",
                suggested_documents=["company_policies", "procedures", "manuals"]
            )

        # Check if question has multiple question marks or is very open-ended
        if question.count("?") > 1 or any(q in question_lower for q in ["não sei", "don't know", "talvez", "maybe"]):
            return RouteDecision(
                route=RouteType.CLARIFY,
                confidence=0.6,
                reasoning="Pergunta parece incerta ou tem múltiplas partes",
                clarifying_questions=[
                    "Vamos focar em um aspecto específico. Qual é sua principal dúvida?"
                ]
            )

        # Default to direct answer for general knowledge questions
        return RouteDecision(
            route=RouteType.DIRECT,
            confidence=0.6,
            reasoning="Pergunta parece ser de conhecimento geral"
        )

    def _llm_based_routing(self, question: str, context: Optional[str] = None) -> RouteDecision:
        """
        Use an LLM to intelligently route the query.
        """
        system_prompt = """Você é um agente de roteamento de consultas. Sua função é analisar uma pergunta e decidir a melhor estratégia para respondê-la:

1. **RAG** (Retrieval-Augmented Generation): Use quando a pergunta:
   - Requer informações específicas da organização/empresa
   - Menciona documentos, políticas, procedimentos internos
   - Pede dados específicos que provavelmente estão em documentos
   - Exemplo: "Qual é a política de férias da empresa?"

2. **DIRECT** (Resposta Direta): Use quando a pergunta:
   - É sobre conhecimento geral que o modelo já possui
   - Não requer informações específicas da organização
   - Pode ser respondida com conhecimento de treinamento do modelo
   - Exemplo: "Como funciona fotossíntese?"

3. **CLARIFY** (Clarificar): Use quando a pergunta:
   - É vaga ou ambígua
   - Falta contexto importante
   - Tem múltiplas interpretações possíveis
   - É muito curta ou genérica
   - Exemplo: "Como faço isso?" (sem especificar o que)

Responda APENAS com um JSON no seguinte formato:
{
    "route": "rag" | "direct" | "clarify",
    "confidence": 0.0-1.0,
    "reasoning": "Explicação da decisão",
    "clarifying_questions": ["pergunta1", "pergunta2"] (apenas se route=clarify),
    "suggested_documents": ["doc1", "doc2"] (apenas se route=rag)
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

            # Parse the result into a RouteDecision
            return RouteDecision(
                route=RouteType(result["route"]),
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                clarifying_questions=result.get("clarifying_questions"),
                suggested_documents=result.get("suggested_documents")
            )

        except Exception as e:
            print(f"Error during LLM routing: {e}")
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
