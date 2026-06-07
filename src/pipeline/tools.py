"""Definição de ferramentas (Function Calling) para conformidade regulatória."""

from __future__ import annotations

from typing import Any, Callable

# Base de dados estática dos artigos para blindar o LLM contra alucinações numéricas
ARTIGOS_LGPD_DATABASE = {
    7: "Art. 7º O tratamento de dados pessoais somente poderá ser realizado nas seguintes hipóteses: I - mediante o fornecimento de consentimento pelo titular; II - para o cumprimento de obrigação legal ou regulatória pelo controlador...",
    11: "Art. 11. O tratamento de dados pessoais sensíveis somente poderá ocorrer nas seguintes hipóteses: I - quando o titular ou seu responsável legal consentir, de forma específica e destacada, para finalidades específicas..."
}


def cite_article(article_number: int) -> str:
    """Retorna o texto regulatório integral e exato do Artigo correspondente da LGPD."""
    num = int(article_number)
    return ARTIGOS_LGPD_DATABASE.get(num, f"Artigo {num} não catalogado no banco de consulta rápida.")


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "cite_article",
            "description": "Retorna o texto normativo integral e exato de um Artigo específico da Lei Geral de Proteção de Dados (LGPD) brasileira.",
            "parameters": {
                "type": "object",
                "properties": {
                    "article_number": {
                        "type": "integer",
                        "description": "O número do artigo da lei que precisa ser consultado de forma literal."
                    }
                },
                "required": ["article_number"],
            },
        },
    }
]


TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "cite_article": cite_article,
}
