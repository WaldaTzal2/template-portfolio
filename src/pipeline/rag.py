def answer(self, question: str, k: int = 5) -> dict[str, Any]:
    """Orquestra a busca, gerencia o cache semântico, executa o roteamento e gera a resposta."""
    if not self.client:
        raise ValueError(
            "Groq Client não inicializado. Verifique a configuração da sua GROQ_API_KEY.")

    # 1. VALIDAÇÃO DE REDUÇÃO DE CUSTO: Consulta ao Cache Semântico
    cached_response = self._check_cache(question)
    if cached_response:
        return cached_response

    # 2. SELEÇÃO DE ESCOPO RÁPIDO (Guarda-reio estrutural)
    perguntas_curtas = ["olá", "oi", "bom dia",
                        "boa tarde", "quem é você", "ajuda"]
    if question.lower().strip() in perguntas_curtas or len(question.strip()) < 12:
        modelo_flash = "llama-3.1-8b-instant"

        response = self.client.chat.completions.create(
            model=modelo_flash,
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA_LGPD},
                {"role": "user", "content": question}
            ]
        )
        resposta_texto = response.choices[0].message.content

        payload = {
            "answer": resposta_texto if resposta_texto else "Não foi possível gerar uma resposta rápida.",
            "sources": [],
            "routing": {
                "model": modelo_flash,
                "complexity": "simple",
                "reason": "Interação inicial simplificada ou saudação tratada pelo modelo Llama 8B."
            }
        }
        self._save_cache(question, payload)
        return payload

    # 3. RECUPERAÇÃO DO RAG (Retrieval)
    hits = self.retrieve(question, k=k)
    contexto_formatado = "\n\n".join(
        [f"--- Trecho [{h['source']}, Pág. {h['page']}]: ---\n{h['text']}" for h in hits]
    )

    # 4. MODEL ROUTING COM TRATAMENTO DE ERRO (Mapeado para modelos estáveis do Groq)
    modelo_escolhido = "llama-3.3-70b-versatile"  # Fallback estável atualizado
    complexidade = "complex"
    motivo = "Análise detalhada de conformidade regulatória."

    try:
        decisao_rota = classify_complexity(question)
        if decisao_rota:
            if hasattr(decisao_rota, "model") or isinstance(decisao_rota, dict):
                nome_modelo = getattr(
                    decisao_rota, "model", "") or decisao_rota.get("model", "")
                if "8b" in nome_modelo.lower() or "flash" in nome_modelo.lower():
                    modelo_escolhido = "llama-3.1-8b-instant"
                    complexidade = "simple"
                else:
                    modelo_escolhido = "llama-3.3-70b-versatile"
                    complexidade = "complex"

                motivo = getattr(decisao_rota, "reason", motivo) if hasattr(
                    decisao_rota, "reason") else decisao_rota.get("reason", motivo)
    except Exception:
        pass

    # 5. MONTAGEM COMPLETA DO PROMPT COM PERSONA, DIRETRIZES E CONTEXTO
    prompt_usuario = (
        f"Por favor, analise a demanda abaixo considerando estritamente as regras de conformidade "
        f"fornecidas no contexto.\n\n"
        f"CONTEXTO DOS DOCUMENTOS INTERNOS:\n{contexto_formatado}\n\n"
        f"PERGUNTA DO USUÁRIO: {question}"
    )

    # 6. GERAÇÃO DA RESPOSTA VIA GROQ (Generation)
    response = self.client.chat.completions.create(
        model=modelo_escolhido,
        messages=[
            {"role": "system", "content": PROMPT_SISTEMA_LGPD},
            {"role": "user", "content": prompt_usuario}
        ]
    )
    resposta_final_texto = response.choices[0].message.content

    # 7. FILTRAGEM E MAPEAMENTO DE FONTES UTILIZADAS (BLINDADO)
    msg_guardrail = "Como seu Assistente de Compliance LGPD, meu escopo de atuação é restrito"
    resposta_final_texto = resposta_final_texto if resposta_final_texto else "Não foi possível gerar uma resposta."

    lista_hits = hits if hits is not None else []

    if msg_guardrail in resposta_final_texto:
        fontes_unicas = []
    else:
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
