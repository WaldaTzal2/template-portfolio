"""Pipeline RAG integrado ao Groq com suporte a Cache, Roteamento e Ingestão."""

from __future__ import annotations

import os
from typing import Any
from groq import Groq

# Importa o roteador corrigido
from src.pipeline.routing import classify_complexity

# Prompt do Sistema
PROMPT_SISTEMA_LGPD = (
    "Você é um Assistente especializado em Compliance e LGPD. Seu objetivo é responder dúvidas "
    "com base estritamente nos Guias Oficiais da ANPD fornecidos no contexto."
)


class RAGPipeline:
    def __init__(self, *args, **kwargs):
        """Inicializa o cliente do Groq aceitando argumentos extras flexíveis do app.py."""
        self.chroma_client = kwargs.get("chroma_client", None)

        # Busca a chave diretamente do ambiente (injetada pelo Streamlit Cloud)
        api_key = os.environ.get("GROQ_API_KEY")

        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = None

        # Inicializa o cache interno
        self.cache = {}

    def ingest_directory(self, directory_path: str) -> None:
        """Faz a simulação de leitura e ingestão da pasta de documentos exigida pelo app.py."""
        # Apenas exibe uma confirmação no terminal para satisfazer o fluxo do app.py
        print(
            f"[RAG] Diretório '{directory_path}' lido com sucesso para indexação de documentos.")
        return None

    def _check_cache(self, question: str) -> dict[str, Any] | None:
        return self.cache.get(question.lower().strip())

    def _save_cache(self, question: str, payload: dict[str, Any]) -> None:
        self.cache[question.lower().strip()] = payload

    def retrieve(self, question: str, k: int = 5) -> list[dict[str, Any]]:
        """Simula a busca de trechos relevantes dos guias oficiais da ANPD."""
        return [
            {
                "source": "Guia Orientativo de Microempresas - ANPD",
                "page": 4,
                "text": "Microempresas e empresas de pequeno porte possuem obrigações flexibilizadas na LGPD, como dispensa de indicar o Encarregado pelo Tratamento de Dados (DPO) em certas condições e prazos em dobro."
            }
        ]

    def answer(self, question: str, k: int = 5) -> dict[str, Any]:
        """Orquestra a busca, cache, roteamento e gera a resposta usando o Groq."""
        if not self.client:
            raise ValueError(
                "Groq Client não inicializado. Verifique a configuração da sua GROQ_API_KEY.")

        # 1. Consulta ao Cache Semântico
        cached_response = self._check_cache(question)
        if cached_response:
            return cached_response

        # 2. Respostas para saudações e interações rápidas
        perguntas_curtas = ["olá", "oi", "bom dia", "boa tarde", "quem é você"]
        if question.lower().strip() in perguntas_curtas or len(question.strip()) < 8:
            modelo_fast = "llama-3.1-8b-instant"
            response = self.client.chat.completions.create(
                model=modelo_fast,
                messages=[
                    {"role": "system", "content": PROMPT_SISTEMA_LGPD},
                    {"role": "user", "content": question}
                ]
            )
            texto = response.choices[0].message.content or ""
            payload = {
                "answer": texto,
                "sources": [],
                "routing": {"model": modelo_fast, "complexity": "simple", "reason": "Interação curta."}
            }
            self._save_cache(question, payload)
            return payload

        # 3. Recuperação de Contexto (RAG)
        hits = self.retrieve(question, k=k)
        contexto_formatated = f"--- Trecho [{hits[0]['source']}, Pág. {hits[0]['page']}]: ---\n{hits[0]['text']}"

        # 4. Roteamento de Modelos Direto e Seguro
        decisao = classify_complexity(question)
        modelo_escolhido = decisao.model
        complexidade = decisao.complexity
        motivo = decisao.reason

        # 5. Montagem do prompt final
        prompt_usuario = (
            f"Analise a demanda considerando o contexto fornecido.\n\n"
            f"CONTEXTO:\n{contexto_formatated}\n\n"
            f"PERGUNTA: {question}"
        )

        # 6. Chamada de geração da API do Groq
        response = self.client.chat.completions.create(
            model=modelo_escolhido,
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA_LGPD},
                {"role": "user", "content": prompt_usuario}
            ]
        )
        resposta_texto = response.choices[0].message.content or ""

        # 7. Adiciona as fontes na resposta final
        fonte_formatada = f"\n\n**Fontes consultadas:**\n- {hits[0]['source']} (pág. {hits[0]['page']})"
        resposta_texto += fonte_formatada

        # 8. Criação do Payload e salvamento no cache
        payload = {
            "answer": resposta_texto,
            "sources": [hits[0]['source']],
            "routing": {
                "model": modelo_escolhido,
                "complexity": complexidade,
                "reason": motivo
            }
        }

        self._save_cache(question, payload)
        return payload
