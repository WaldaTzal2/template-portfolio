"""Roteador de modelos baseado na complexidade da pergunta."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class RouteDecision:
    model: str
    complexity: str
    reason: str


def classify_complexity(query: str) -> RouteDecision:
    """Classifica complexidade da query para escolher modelo (cheap vs premium) do Groq."""
    # Configura os modelos oficiais e equivalentes do Groq
    cheap_model = os.environ.get("CHEAP_MODEL", "llama3-8b-8192")
    premium_model = os.environ.get("PREMIUM_MODEL", "llama3-70b-8192")

    query_lower = query.lower()
    gatilhos_complexos = ["explique", "compare",
                          "analise", "projete", "audite", "vulnerabilidade"]

    # Heurística para query curta e direta
    if len(query) < 60 and query.strip().endswith("?"):
        return RouteDecision(
            model=cheap_model,
            version_type="groq",  # Garante marcação interna se necessário
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
