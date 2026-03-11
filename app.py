import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st
import uuid
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from rag.busca import ESCOPO_CLOUD, ESCOPO_ON_PREMISE, buscar_contexto
from rag.qa_cache import (
    buscar_resposta_exata,
    compactar_cache,
    inicializar_cache,
    normalizar_pergunta,
    salvar_resposta,
)
from rag.web_fetch import fetch_url_text
from rag.web_search import search_searxng
from rag.youtube_transcript import fetch_youtube_transcript

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# =========================
# CONFIG INICIAL
# =========================

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR_ENV = os.getenv("IASMIN_DATA_DIR", "").strip()
DATA_DIR = Path(DATA_DIR_ENV).expanduser() if DATA_DIR_ENV else ROOT_DIR

def _resolver_cache_dir() -> Path:
    env_dir = os.getenv("IASMIN_CACHE_DIR", "").strip()
    if env_dir:
        base = Path(env_dir).expanduser()
    else:
        base = DATA_DIR / ".cache"
    try:
        base.mkdir(parents=True, exist_ok=True)
        teste = base / ".write_test"
        teste.write_text("ok", encoding="utf-8")
        teste.unlink(missing_ok=True)
        return base
    except Exception:
        tmp = Path(os.getenv("TMPDIR", "/tmp")) / "iasmin"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp

CACHE_DIR = _resolver_cache_dir()
CACHE_DB_PATH = str(CACHE_DIR / "qa_cache.sqlite")
CACHE_SCOPE_DEFAULT = "nao_definido"
SEARXNG_URL = os.getenv("SEARXNG_URL", "").strip()
WEB_SEARCH_ENABLED = bool(SEARXNG_URL)
WEB_MAX_RESULTS = int(os.getenv("WEB_MAX_RESULTS", "5"))
WEB_MAX_CHARS = int(os.getenv("WEB_MAX_CHARS", "6000"))


def assinatura_indice() -> str:
    faiss_path = DATA_DIR / "rag" / "base_vetorial" / "index.faiss"
    pkl_path = DATA_DIR / "rag" / "base_vetorial" / "index.pkl"
    partes = []
    for p in (faiss_path, pkl_path):
        if p.exists():
            stt = p.stat()
            partes.append(f"{p.name}:{stt.st_mtime_ns}:{stt.st_size}")
        else:
            partes.append(f"{p.name}:missing")
    return hashlib.sha256("|".join(partes).encode("utf-8")).hexdigest()[:16]


def assinatura_web() -> str:
    data = datetime.utcnow().strftime("%Y-%m-%d")
    return f"web:{data}"

if sys.version_info >= (3, 14):
    st.error(
        "Python 3.14 detectado na venv atual. Este projeto requer Python 3.11/3.12 "
        "para estabilidade com LangChain/FAISS."
    )
    st.stop()


def obter_api_key() -> str:
    # Prioriza variável de ambiente e usa Streamlit secrets como fallback.
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "").strip()
    except Exception:
        api_key = ""
    return api_key


st.set_page_config(
    page_title="IAsmiN, a consultora SAP IA da Nova",
    page_icon="assets/logo_nova.png",
    layout="wide"
)

# =========================
# CACHE DE PERSONA
# =========================

@st.cache_data
def carregar_persona():
    try:
        with open("persona.txt", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "Você é uma consultora SAP EPR Cloud da Nova Consulting."


# =========================
# MEMÓRIA POR SESSÃO
# =========================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "historico" not in st.session_state:
    st.session_state.historico = ChatMessageHistory()

if "sap_escopo" not in st.session_state:
    st.session_state.sap_escopo = None

if "pergunta_pendente" not in st.session_state:
    st.session_state.pergunta_pendente = None


def get_session_history(session_id):
    return st.session_state.historico


# =========================
# CACHE DO LLM
# =========================

@st.cache_resource
def carregar_llm():
    api_key = obter_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY ausente")
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        streaming=True,
        api_key=api_key
    )

try:
    llm = carregar_llm()
except RuntimeError:
    st.error(
        "OPENAI_API_KEY não configurada. Defina a variável de ambiente "
        "`OPENAI_API_KEY` (ou `st.secrets`) e reinicie o app."
    )
    st.stop()

# =========================
# PROMPT
# =========================

persona = carregar_persona()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", persona + """

Você possui acesso a um banco de conhecimento abaixo:

{contexto}

Nível de confiança de recuperação (0 a 1): {confianca}
Fontes recuperadas: {fontes}
Escopo SAP definido: {escopo_sap}

Regras de resposta:
1) Priorize o contexto SAP recuperado.
1.1) Use apenas fontes aderentes ao escopo SAP definido.
2) Se a confiança for baixa (<0.20), peça mais detalhes para responder com precisão.
3) Não invente dados técnicos (transações, tabelas, notas, configurações).
4) Para cada afirmação técnica relevante, use citação inline [1], [2], etc.
5) Ao final, inclua obrigatoriamente a seção "Fontes" numerada.
"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)
inicializar_cache(CACHE_DB_PATH)
compactar_cache(CACHE_DB_PATH, manter_registros=5000)

# =========================
# CHAIN
# =========================

chain = prompt | llm

chain_com_memoria = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# =========================
# HISTÓRICO
# =========================

from langchain_core.messages import HumanMessage


LIMIAR_CONFIANCA = 0.08
PERGUNTA_ESCOPO = "Qual versão é o seu SAP? On premise ou Public?"


def _resposta_baixa_confianca(fontes: list[str]) -> str:
    referencias = (
        "\n".join(f"{idx}. {fonte}" for idx, fonte in enumerate(fontes, start=1))
        if fontes
        else "1. Nenhuma fonte SAP suficientemente relevante foi recuperada."
    )
    return (
        "Não encontrei evidência suficiente na base SAP Help para responder com precisão.\n\n"
        "Para eu te orientar melhor, envie mais contexto (módulo SAP, processo exato, "
        "mensagem de erro, versão/escopo cloud ou on-premise).\n\n"
        "Fontes\n"
        f"{referencias}"
    )


def _garantir_secao_fontes(texto: str, fontes: list[str]) -> str:
    if "fontes" in texto.lower():
        return texto
    referencias = (
        "\n".join(f"{idx}. {fonte}" for idx, fonte in enumerate(fontes, start=1))
        if fontes
        else "1. Sem fontes recuperadas."
    )
    return f"{texto}\n\nFontes\n{referencias}"


def _classificar_escopo(resposta: str) -> str | None:
    r = resposta.lower()
    tokens_on_prem = ["on premise", "on-prem", "onprem", "ecc", "ehp", "rise"]
    tokens_cloud = ["public", "grow", "s/4hana cloud", "cloud public"]
    if any(t in r for t in tokens_on_prem):
        return ESCOPO_ON_PREMISE
    if any(t in r for t in tokens_cloud):
        return ESCOPO_CLOUD
    return None


def _nome_escopo(escopo: str | None) -> str:
    if escopo == ESCOPO_ON_PREMISE:
        return "SAP S/4HANA On-Premise"
    if escopo == ESCOPO_CLOUD:
        return "SAP S/4HANA Cloud Public"
    return "Não definido"


def _eh_comando_trocar_escopo(texto: str) -> bool:
    t = texto.strip().lower()
    comandos = {
        "trocar versão sap",
        "trocar versao sap",
        "mudar versão sap",
        "mudar versao sap",
        "/trocar-versao-sap",
    }
    return t in comandos


def _eh_comando_bbp(texto: str) -> bool:
    t = texto.strip().lower()
    comandos = {
        "gerar bbp",
        "bbp",
        "/bbp",
        "template bbp",
        "/bbp-template",
    }
    return t in comandos


def _carregar_template_bbp() -> str:
    caminho = ROOT_DIR / "templates" / "bbp_template.md"
    try:
        return caminho.read_text(encoding="utf-8")
    except Exception:
        return "Template BBP indisponível."

def _logo_header() -> str:
    if Path("assets/logo_nova.png").exists():
        return "assets/logo_nova.png"
    if Path("assets/logocasachao.png").exists():
        return "assets/logocasachao.png"
    return "assets/user_3.png"


def _avatar_agente() -> str:
    if Path("assets/iasmin.png").exists():
        return "assets/iasmin.png"
    if Path("assets/user_3.png").exists():
        return "assets/user_3.png"
    return "assets/logo_sap.png"


def _montar_contexto_web(pergunta: str, escopo_sap: str | None) -> tuple[str, list[str]]:
    if not WEB_SEARCH_ENABLED:
        return "", []

    escopo_term = "SAP S/4HANA"
    if escopo_sap == ESCOPO_ON_PREMISE:
        escopo_term = "SAP S/4HANA On-Premise"
    elif escopo_sap == ESCOPO_CLOUD:
        escopo_term = "SAP S/4HANA Cloud Public"

    query = f"{pergunta} {escopo_term}"
    dominios = [
        "help.sap.com",
        "community.sap.com",
        "stackoverflow.com",
        "youtube.com",
        "youtu.be",
    ]

    results = search_searxng(
        query=query,
        searxng_url=SEARXNG_URL,
        domains=dominios,
        max_results=WEB_MAX_RESULTS,
    )

    fontes = []
    trechos = []
    for idx, r in enumerate(results, start=1):
        url = r.get("url") or ""
        if not url:
            continue
        fontes.append(url)
        snippet = r.get("content") or r.get("title") or ""
        texto = ""
        if "youtube.com" in url or "youtu.be" in url:
            texto = fetch_youtube_transcript(url, max_chars=WEB_MAX_CHARS) or snippet
        else:
            texto = fetch_url_text(url, max_chars=WEB_MAX_CHARS) or snippet
        if texto:
            trechos.append(f"[Web {idx} | Fonte: {url}]\n{texto}")

    return "\n\n".join(trechos), fontes


# =========================
# HEADER
# =========================

col1, col2 = st.columns([1, 6])

with col1:
    st.image(_logo_header(), width=80)

with col2:
    st.header("Olá, eu sou a IAsmin", divider=True)
    st.caption("Sua consultora SAP virtual")


@st.cache_data(ttl=1800, show_spinner=False)
def _buscar_contexto_cacheado(
    pergunta: str, escopo_sap: str | None, indice_sig: str
) -> dict:
    # indice_sig entra na chave do cache para invalidar quando a base vetorial muda.
    _ = indice_sig
    return buscar_contexto(pergunta, escopo_sap=escopo_sap)

for msg in st.session_state.historico.messages:
    if isinstance(msg, HumanMessage):
        role = "user"
        avatar = "assets/logo_sap.png"
    else:
        role = "assistant"
        avatar = _avatar_agente()

    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.content)

# =========================
# INPUT
# =========================

pergunta = st.chat_input("Pergunte sobre o SAP ERP...")

if pergunta:

    with st.chat_message("user", avatar="assets/logo_sap.png"):
        st.markdown(pergunta)

    with st.chat_message("assistant", avatar=_avatar_agente()):

        resposta_container = st.empty()
        resposta_final = ""

        try:
            retorno_antecipado = False
            if _eh_comando_trocar_escopo(pergunta):
                st.session_state.sap_escopo = None
                st.session_state.pergunta_pendente = None
                resposta_final = (
                    "Versão SAP resetada com sucesso.\n\n"
                    f"{PERGUNTA_ESCOPO}"
                )
                resposta_container.markdown(resposta_final)
                retorno_antecipado = True
            elif _eh_comando_bbp(pergunta):
                st.image(_logo_header(), width=120)
                resposta_final = _carregar_template_bbp()
                resposta_container.markdown(resposta_final)
                retorno_antecipado = True

            pergunta_para_responder = None

            if not retorno_antecipado and st.session_state.sap_escopo is None:
                if st.session_state.pergunta_pendente is None:
                    st.session_state.pergunta_pendente = pergunta
                    resposta_final = PERGUNTA_ESCOPO
                    resposta_container.markdown(resposta_final)
                else:
                    escopo = _classificar_escopo(pergunta)
                    if escopo is None:
                        resposta_final = (
                            "Não identifiquei a versão. Responda com uma destas opções:\n"
                            "- On premise / ECC / EHP / RISE\n"
                            "- Public / GROW"
                        )
                        resposta_container.markdown(resposta_final)
                    else:
                        st.session_state.sap_escopo = escopo
                        pergunta_para_responder = st.session_state.pergunta_pendente
                        st.session_state.pergunta_pendente = None
                        msg_escopo = f"Entendido. Vou considerar o escopo: `{_nome_escopo(escopo)}`."
                        resposta_container.markdown(msg_escopo)
            elif not retorno_antecipado:
                pergunta_para_responder = pergunta

            if not retorno_antecipado and pergunta_para_responder:
                indice_sig = assinatura_indice()
                scope_key = st.session_state.sap_escopo or CACHE_SCOPE_DEFAULT
                pergunta_norm = normalizar_pergunta(pergunta_para_responder)
                cache_hit = buscar_resposta_exata(
                    CACHE_DB_PATH, scope_key, pergunta_norm, indice_sig
                )

                cache_hit_usado = cache_hit is not None
                if cache_hit_usado:
                    resposta_final = cache_hit.answer
                    resposta_container.markdown(resposta_final)
                    if cache_hit.sources:
                        with st.expander("Fontes recuperadas no RAG"):
                            for fonte in cache_hit.sources:
                                st.markdown(f"- {fonte}")
                            st.caption(
                                f"Confiança média da recuperação: {cache_hit.confidence}"
                            )
                else:
                    resultado_rag = _buscar_contexto_cacheado(
                        pergunta_para_responder,
                        escopo_sap=st.session_state.sap_escopo,
                        indice_sig=indice_sig,
                    )
                    contexto_rag = resultado_rag["contexto"]
                    fontes_rag = ", ".join(resultado_rag["fontes"]) if resultado_rag["fontes"] else "Sem fontes"
                    confianca_rag = resultado_rag["confianca"]

                    if confianca_rag < LIMIAR_CONFIANCA and not resultado_rag["fontes"]:
                        web_sig = assinatura_web()
                        web_cache = buscar_resposta_exata(
                            CACHE_DB_PATH, scope_key, pergunta_norm, web_sig
                        )
                        if web_cache is not None:
                            resposta_final = web_cache.answer
                            resposta_container.markdown(resposta_final)
                            if web_cache.sources:
                                with st.expander("Fontes recuperadas no RAG"):
                                    for fonte in web_cache.sources:
                                        st.markdown(f"- {fonte}")
                                    st.caption(
                                        f"Confiança média da recuperação: {web_cache.confidence}"
                                    )
                        else:
                            contexto_web, fontes_web = _montar_contexto_web(
                                pergunta_para_responder,
                                escopo_sap=st.session_state.sap_escopo,
                            )
                            if contexto_web:
                                respostas_fontes = ", ".join(fontes_web) if fontes_web else "Sem fontes"
                                resposta_final = ""
                                for chunk in chain_com_memoria.stream(
                                    {
                                        "input": pergunta_para_responder,
                                        "contexto": contexto_web,
                                        "fontes": respostas_fontes,
                                        "confianca": 0.25,
                                        "escopo_sap": _nome_escopo(st.session_state.sap_escopo),
                                    },
                                    config={"configurable": {"session_id": st.session_state.session_id}}
                                ):
                                    resposta_final += chunk.content
                                    resposta_container.markdown(resposta_final)

                                resposta_final = _garantir_secao_fontes(resposta_final, fontes_web)
                                resposta_container.markdown(resposta_final)
                                salvar_resposta(
                                    CACHE_DB_PATH,
                                    scope=scope_key,
                                    question_raw=pergunta_para_responder,
                                    answer=resposta_final,
                                    sources=fontes_web,
                                    confidence=0.25,
                                    index_signature=web_sig,
                                )
                            else:
                                resposta_final = _resposta_baixa_confianca(resultado_rag["fontes"])
                                resposta_container.markdown(resposta_final)
                    else:
                        resposta_final = ""
                        for chunk in chain_com_memoria.stream(
                            {
                                "input": pergunta_para_responder,
                                "contexto": contexto_rag,
                                "fontes": fontes_rag,
                                "confianca": confianca_rag,
                                "escopo_sap": _nome_escopo(st.session_state.sap_escopo),
                            },
                            config={"configurable": {"session_id": st.session_state.session_id}}
                        ):
                            resposta_final += chunk.content
                            resposta_container.markdown(resposta_final)

                        resposta_final = _garantir_secao_fontes(resposta_final, resultado_rag["fontes"])
                        resposta_container.markdown(resposta_final)
                        salvar_resposta(
                            CACHE_DB_PATH,
                            scope=scope_key,
                            question_raw=pergunta_para_responder,
                            answer=resposta_final,
                            sources=resultado_rag["fontes"],
                            confidence=confianca_rag,
                            index_signature=indice_sig,
                        )

                    if resultado_rag["fontes"]:
                        with st.expander("Fontes recuperadas no RAG"):
                            for fonte in resultado_rag["fontes"]:
                                st.markdown(f"- {fonte}")
                            st.caption(f"Confiança média da recuperação: {confianca_rag}")

        except Exception:
            st.error("⚠️ Erro ao gerar resposta. Tente novamente.")
