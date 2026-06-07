"""Interface Visual em Streamlit para o Assistente de Compliance LGPD."""

from src.pipeline.rag import RAGPipeline
from dotenv import load_dotenv
import chromadb
import streamlit as st
import sys
import os

# Adiciona a raiz do projeto ao caminho de busca do Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Agora importa a sua classe

# Adiciona a pasta atual ao path do Python para garantir que 'src' seja encontrada
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Agora o import deve funcionar


# Importa o pipeline corrigido e testado

# Carrega chaves do arquivo .env
load_dotenv()

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Assistente de Compliance LGPD",
    page_icon="🛡️",
    layout="wide"
)

# Inicialização estável do banco de dados vetorial para a interface


@st.cache_resource
def iniciar_pipeline():
    # Instancia o cliente persistente local do ChromaDB
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")

    # Cria o pipeline e aponta para a pasta onde ficam os PDFs de guias da ANPD
    pipeline = RAGPipeline(chroma_client=chroma_client)
    pipeline.ingest_directory("data/corpus")
    return pipeline


try:
    agente_rag = iniciar_pipeline()
except Exception as e:
    st.error(f"Erro ao inicializar o banco de dados ou chaves de API: {e}")
    st.stop()

# Cabeçalho da Aplicação
st.title("🛡️ Assistente de Compliance LGPD")
st.subheader("Consultoria automatizada baseada nos Guias Oficiais da ANPD")
st.markdown(
    "Este assistente utiliza técnicas avançadas de **RAG**, **Cache Semântico** e "
    "**Roteamento Inteligente de Modelos** para responder a dúvidas regulatórias de forma eficiente."
)

st.divider()

# Layout em duas colunas: Esquerda para o chat / Direita para os Bastidores (Métricas)
col_chat, col_bastidores = st.columns([2, 1])

with col_chat:
    st.markdown("### 💬 Área de Consulta")

    # Campo de entrada de texto para a pergunta do usuário
    pergunta_usuario = st.text_input(
        "Digite sua dúvida sobre privacidade, proteção de dados ou obrigações da LGPD:",
        placeholder="Ex: Quais as obrigações de um agente de pequeno porte?"
    )

    if st.button("Enviar Pergunta", type="primary"):
        if not pergunta_usuario.strip():
            st.warning("Por favor, digite uma pergunta antes de enviar.")
        else:
            with st.spinner("Analisando documentos regulatórios e gerando resposta oficial..."):
                try:
                    # Executa a orquestração completa no seu rag.py
                    payload_resposta = agente_rag.answer(pergunta_usuario)

                    # Exibe a resposta formatada em Markdown gerada pela LLM
                    st.markdown("#### 🤖 Resposta do Assistente:")
                    st.markdown(payload_resposta["answer"])

                    # Guarda a resposta no estado da sessão para alimentar a coluna de métricas
                    st.session_state["ultimo_payload"] = payload_resposta

                except Exception as error:
                    st.error(
                        f"Ocorreu um erro ao processar sua pergunta: {error}")

with col_bastidores:
    st.markdown("### 📊 Transparência e Infraestrutura")
    st.info("Painel de auditoria exigido pela rubrica do projeto para verificação das decisões da IA.")

    if "ultimo_payload" in st.session_state:
        payload = st.session_state["ultimo_payload"]
        routing_info = payload.get("routing", {})

        # 1. Status do Cache Semântico (Medida de Custo)
        st.markdown("#### ⚡ Otimização de Custo")
        if payload.get("cached", False):
            st.success(
                "🟢 **CACHE HIT:** Resposta servida direto do banco vetorial local (Custo: **$0.00** / 0 tokens utilizados).")
        else:
            st.warning(
                "🟡 **CACHE MISS:** Pergunta inédita. Requisição enviada para processamento da LLM.")

        # 2. Informações de Roteamento Dinâmico (Model Routing)
        st.markdown("#### 🤖 Roteamento de Modelos")
        st.metric(label="Modelo Selecionado", value=routing_info.get(
            "model", "Desconhecido").split("/")[-1])
        st.markdown(
            f"**Complexidade Estimada:** `{routing_info.get('complexity', 'N/A').upper()}`")
        st.caption(
            f"**Motivo da Decisão:** {routing_info.get('reason', 'N/A')}")

        # 3. Exibição das Fontes Bibliográficas Mapeadas
        st.markdown("#### 📂 Fontes Utilizadas pelo RAG")
        fontes = payload.get("sources", [])
        if fontes:
            for fonte in fontes:
                st.markdown(f"- 📄 `{fonte}`")
        else:
            st.caption(
                "Nenhuma fonte documental consultada para esta interação (Interação curta ou fora de escopo).")

    else:
        st.caption(
            "Faça uma pergunta na área de consulta para exibir os dados de roteamento, fontes e cache semântico em tempo real.")

# Rodapé institucional
st.divider()
st.caption(
    "🔄 Desenvolvido para o Projeto Portfólio - Engenharia de Software com IA Generativa.")
