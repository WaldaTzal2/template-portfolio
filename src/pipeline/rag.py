"""Pipeline RAG (Retrieval-Augmented Generation) para Compliance e Proteção de Dados."""

from __future__ import annotations

import os
import json
import hashlib
from typing import Any
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.pipeline.routing import classify_complexity

# Configuração simples para silenciar avisos do gRPC
os.environ["GRPC_VERBOSITY"] = "ERROR"

# =====================================================================
# PASSO 3: CONFIGURAÇÃO DE ENGENHARIA DE PROMPTS (PERSONA & GUARDA-REIOS)
# =====================================================================
PROMPT_SISTEMA_LGPD = (
    "Você é o Assistente Especialista em Compliance da LGPD, um consultor técnico e analista especializado "
    "na Lei Geral de Proteção de Dados (Lei nº 13.709/2018) e nas diretrizes oficiais da ANPD.\n"
    "Sua missão é guiar encarregados de dados (DPOs) e pequenas empresas de forma didática, clara e profissional.\n\n"
    "[DIRETRIZES DE COMPORTAMENTO E CONFIGURAÇÃO DA SAÍDA]\n"
    "1. Responda com um tom estritamente profissional, corporativo e seguro.\n"
    "2. Baseie suas respostas estritamente nas informações fornecidas no contexto extraído dos guias oficiais.\n"
    "3. Formate a saída obrigatoriamente utilizando Markdown rico: termos e conceitos jurídicos importantes em **negrito**, "
    "e listas com marcadores (bullet points) claros para deveres, passos ou penalidades.\n"
    "4. Se o contexto fornecido pelo RAG não contiver dados suficientes para responder, admita de forma transparente "
    "que não localizou a resposta nos guias oficiais indexados. Nunca alucine ou invente artigos da lei.\n\n"
    "[GUARDA-REIOS / CONTROLE DE FORA DE ESCOPO]\n"
    "Se o usuário fizer qualquer tipo de pergunta que NÃO envolva privacidade, proteção de dados pessoais, conformidade regulatória, "
    "segurança da informação básica ou LGPD (como receitas, códigos de programação genéricos, futebol, fofocas ou piadas), você DEVE ignorar "
    "o contexto e responder estritamente com a mensagem padrão de bloqueio abaixo:\n"
    "\"Como seu Assistente de Compliance LGPD, meu escopo de atuação é restrito exclusivamente a dúvidas sobre proteção de dados, "
    "privacidade e conformidade legal. Por favor, reformule sua pergunta focando nesses temas.\"\n\n"
    "[AVISO DE ISENÇÃO DE RESPONSABILIDADE]\n"
    "Ao final de cada resposta instrutiva, adicione uma linha horizontal separadora (---) e inclua textualmente o seguinte aviso legal:\n"
    "*Aviso: Esta orientação possui caráter puramente educativo e informativo com base nos guias da ANPD e na legislação vigente, "
    "não substituindo uma assessoria ou parecer jurídico formal específico para o seu negócio.*"
)


class RAGPipeline:
    def __init__(self, chroma_client: Any):
        """Inicializa a pipeline com o cliente ChromaDB injetado."""
        self.chroma = chroma_client

        # Garante a existência da collection limpa para os documentos do RAG
        try:
            self.chroma.delete_collection("docs")
        except Exception:
            pass

        self.collection = self.chroma.create_collection(name="docs")

        # Inicializa ou recupera a coleção dedicada ao CACHE SEMÂNTICO (Redução de Custos)
        try:
            self.cache_collection = self.chroma.get_or_create_collection(
                name="cache_semantico")
        except Exception:
            self.cache_collection = self.chroma.create_collection(
                name="cache_semantico")

        # Configura a API do Gemini
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)

    def ingest_directory(self, target_dir: str) -> None:
        """Carrega PDFs, quebra em pedaços (chunks) e indexa no ChromaDB."""
        if not os.path.exists(target_dir) or not os.listdir(target_dir):
            return

        # 1. Carrega os PDFs da pasta
        loader = PyPDFDirectoryLoader(target_dir)
        documents = loader.load()

        if not documents:
            return

        # 2. Divide em pedaços menores (chunks) mantendo integridade textual
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)

        # 3. Prepara os dados no formato que o ChromaDB exige
        documents_text = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                "source": os.path.basename(chunk.metadata.get("source", "unknown")),
                "page": int(chunk.metadata.get("page", 0)) + 1
            }
            for chunk in chunks
        ]
        ids = [f"doc_{i}" for i in range(len(chunks))]

        # 4. Salva no banco vetorial local
        if documents_text:
            self.collection.add(
                documents=documents_text,
                metadatas=metadatas,
                ids=ids
            )

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Busca os top-k chunks mais similares à pergunta do usuário."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )

        hits = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [
                {}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [
                0.0] * len(docs)

            for idx in range(len(docs)):
                hits.append({
                    "text": docs[idx],
                    "source": metas[idx].get("source", "unknown"),
                    "page": metas[idx].get("page", 0),
                    "distance": distances[idx]
                })
        return hits

    def _check_cache(self, query: str) -> dict[str, Any] | None:
        """Verifica se existe uma resposta pré-existente semanticamente semelhante no cache."""
        if not self.cache_collection or self.cache_collection.count() == 0:
            return None

        cache_results = self.cache_collection.query(
            query_texts=[query],
            n_results=1
        )

        if cache_results and cache_results.get("documents") and cache_results["documents"][0]:
            distance = cache_results["distances"][0][0] if cache_results.get(
                "distances") else 1.0

            if distance < 0.28:
                try:
                    cached_data = json.loads(cache_results["documents"][0][0])
                    cached_data["cached"] = True
                    return cached_data
                except Exception:
                    return None
        return None

    def _save_cache(self, query: str, answer_payload: dict[str, Any]) -> None:
        """Salva a resposta consolidada estruturada no banco vetorial de cache semântico."""
        id_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
        serialized_payload = json.dumps(answer_payload, ensure_ascii=False)

        self.cache_collection.add(
            documents=[serialized_payload],
            ids=[f"cache_{id_hash}"]
        )

    def answer(self, question: str, k: int = 5) -> dict[str, Any]:
        """Orquestra a busca, gerencia o cache semântico, executa o roteamento e gera a resposta."""

        # 1. VALIDAÇÃO DE REDUÇÃO DE CUSTO: Consulta ao Cache Semântico
        cached_response = self._check_cache(question)
        if cached_response:
            return cached_response

        # 2. SELEÇÃO DE ESCOPO RÁPIDO (Guarda-reio estrutural)
        perguntas_curtas = ["olá", "oi", "bom dia",
                            "boa tarde", "quem é você", "ajuda"]
        if question.lower().strip() in perguntas_curtas or len(question.strip()) < 12:
            modelo_flash = "models/gemini-1.5-flash"
            model = genai.GenerativeModel(
                model_name=modelo_flash, system_instruction=PROMPT_SISTEMA_LGPD)
            response = model.generate_content(question)

            payload = {
                "answer": response.text if response else "Não foi possível gerar uma resposta rápida.",
                "sources": [],
                "routing": {
                    "model": modelo_flash,
                    "complexity": "simple",
                    "reason": "Interação inicial simplificada ou saudação tratada pelo modelo Flash."
                }
            }
            self._save_cache(question, payload)
            return payload

        # 3. RECUPERAÇÃO DO RAG (Retrieval)
        hits = self.retrieve(question, k=k)
        contexto_formatado = "\n\n".join(
            [f"--- Trecho [{h['source']}, Pág. {h['page']}]: ---\n{h['text']}" for h in hits]
        )

        # 4. MODEL ROUTING COM TRATAMENTO DE ERRO (BLINDAGEM DO PROMPT)
        modelo_escolhido = "models/gemini-1.5-pro"  # Fallback padrão robusto
        complexidade = "complex"
        motivo = "Análise detalhada de conformidade regulatória."

        try:
            decisao_rota = classify_complexity(question)
            if decisao_rota:
                # Verifica se o retorno é um objeto com atributo ou um dicionário
                if hasattr(decisao_rota, "model"):
                    modelo_escolhido = decisao_rota.model
                    complexidade = getattr(
                        decisao_rota, "complexity", "complex")
                    motivo = getattr(decisao_rota, "reason", motivo)
                elif isinstance(decisao_rota, dict):
                    modelo_escolhido = decisao_rota.get(
                        "model", modelo_escolhido)
                    complexidade = decisao_rota.get("complexity", complexidade)
                    motivo = decisao_rota.get("reason", motivo)
        except Exception:
            pass  # Se o arquivo externo falhar, usa o modelo pro para garantir a execução

        # 5. MONTAGEM COMPLETA DO PROMPT COM PERSONA, DIRETRIZES E CONTEXTO
        model = genai.GenerativeModel(
            model_name=modelo_escolhido,
            system_instruction=PROMPT_SISTEMA_LGPD
        )

        prompt_usuario = (
            f"Por favor, analise a demanda abaixo considerando estritamente as regras de conformidade "
            f"fornecidas no contexto.\n\n"
            f"CONTEXTO DOS DOCUMENTOS INTERNOS:\n{contexto_formatado}\n\n"
            f"PERGUNTA DO USUÁRIO: {question}"
        )

        # 6. GERAÇÃO DA RESPOSTA (Generation)
        response = model.generate_content(prompt_usuario)

        # 7. FILTRAGEM E MAPEAMENTO DE FONTES UTILIZADAS (BLINDADO)
        msg_guardrail = "Como seu Assistente de Compliance LGPD, meu escopo de atuação é restrito"
        resposta_final_texto = response.text if response else "Não foi possível gerar uma resposta."

        # Garante que hits é tratado como lista mesmo se for None
        lista_hits = hits if hits is not None else []

        if msg_guardrail in resposta_final_texto:
            fontes_unicas = []
        else:
            # Cria o conjunto de fontes apenas se houver hits
            fontes_unicas = list(
                {f"{h['source']} (pág. {h['page']})" for h in lista_hits})

            if fontes_unicas:
                texto_fontes = "\n\n**Fontes consultadas nos guias oficiais:**\n" + \
                    "\n".join([f"- {f}" for f in fontes_unicas])
                resposta_final_texto += texto_fontes

        # 8. CONSTRUÇÃO DO PAYLOAD FINAL E ATUALIZAÇÃO DO CACHE
        payload = {
            "answer": resposta_final_texto,
            "sources": fontes_unicas,
            "routing": {
                "model": modelo_escolhido,
                "complexity": complexidade,
                "reason": motivo
            }
        }

        self._save_cache(question, payload)

        return payload
streamlit run app.py