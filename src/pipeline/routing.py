"""Roteador de modelos baseado na complexidade da pergunta."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouteDecision:
    model: str
    complexity: str
    reason: str


def classify_complexity(query: str) -> RouteDecision:
    """Classifica complexidade da query para escolher modelo (cheap vs premium) do Groq."""
    # Modelos atualizados e suportados pelo Groq
    cheap_model = "llama-3.1-8b-instant"
    premium_model = "llama-3.3-70b-versatile"

    query_lower = query.lower()
    gatilhos_complexos = ["explique", "compare",
                          "analise", "projete", "audite", "vulnerabilidade"]

    # Heurística para query curta e direta
    if len(query) < 60 and query.strip().endswith("?"):
        return RouteDecision(
            model=cheap_model,
            complexity="simple",
            reason="Query curta finalizada com ponto de interrogação."
        )

    # Heurística para termos que exigem maior raciocínio cognitivo
    if any(palavra in query_lower for palavra in gatilhos_complexos):
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason="Contém palavras-chave de alta complexidade analítica."
        )

    # Padrão (Fallback econômico)
    return RouteDecision(
        model=cheap_model,
        complexity="simple",
        reason="Heurística padrão: direcionado ao modelo econômico do Groq."
    )
