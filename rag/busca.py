import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR_ENV = os.getenv("IASMIN_DATA_DIR", "").strip()
DATA_DIR = Path(DATA_DIR_ENV).expanduser() if DATA_DIR_ENV else ROOT_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_PATH = str(DATA_DIR / "rag" / "base_vetorial")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_CROSS_ENCODER = None
ENABLE_CROSS_ENCODER = os.getenv("ENABLE_CROSS_ENCODER", "0").strip() == "1"
ESCOPO_ON_PREMISE = "on_premise"
ESCOPO_CLOUD = "cloud_public"


def _normalizar_fonte(metadata: Dict[str, Any]) -> str:
    source = metadata.get("source", "")
    page = metadata.get("page")
    if source and page is not None:
        return f"{source} (p. {page + 1})"
    return source or "fonte_indefinida"


def _score_distancia_para_confianca(score_distancia: float) -> float:
    return 1.0 / (1.0 + float(score_distancia))


def _score_rerank_para_confianca(score_rerank: float) -> float:
    # Normalização aproximada para produzir faixa de 0 a 1.
    return 1.0 / (1.0 + pow(2.718281828, -float(score_rerank)))


def _tokenizar(texto: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", texto.lower()))


def _normalizar_texto(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )


def _rerank_lexical(pergunta: str, textos: List[str]) -> List[float]:
    tokens_pergunta = _tokenizar(pergunta)
    scores = []
    for texto in textos:
        tokens_texto = _tokenizar(texto)
        if not tokens_texto:
            scores.append(0.0)
            continue
        intersecao = len(tokens_pergunta.intersection(tokens_texto))
        scores.append(intersecao / len(tokens_pergunta) if tokens_pergunta else 0.0)
    return scores


def _rerank(pergunta: str, textos: List[str]) -> List[float]:
    global _CROSS_ENCODER
    if not ENABLE_CROSS_ENCODER:
        return _rerank_lexical(pergunta, textos)
    try:
        from sentence_transformers import CrossEncoder

        if _CROSS_ENCODER is None:
            _CROSS_ENCODER = CrossEncoder(RERANK_MODEL)
        pares = [(pergunta, texto) for texto in textos]
        return [float(s) for s in _CROSS_ENCODER.predict(pares)]
    except Exception:
        return _rerank_lexical(pergunta, textos)


def _chave_doc(doc: Any) -> str:
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    return f"{source}::{page}::{doc.page_content[:120]}"


def _expandir_consultas(pergunta: str) -> List[str]:
    q = pergunta.strip()
    q_norm = _normalizar_texto(q)
    consultas = [q]

    # Toda consulta ganha âncora SAP para evitar recuperação "genérica".
    consultas.append(f"{q} SAP S/4HANA")

    mapa = {
        "subcontratacao": ["subcontracting", "external processing", "third-party processing"],
        "compras": ["procurement", "purchasing"],
        "requisicao de compra": ["purchase requisition", "procurement request", "manage purchase requisitions"],
        "pedido de compra": ["purchase order", "create purchase order"],
        "estoque": ["inventory", "stock"],
        "fatura": ["invoice", "billing"],
        "financeiro": ["finance", "fi"],
        "vendas": ["sales", "sd"],
        "producao": ["production", "pp"],
    }

    extras = []
    for termo, termos_en in mapa.items():
        if termo in q_norm:
            extras.extend(termos_en)

    # Se não detectar domínio explícito, força termos centrais SAP para recall.
    if not extras:
        extras.extend(["sap", "s/4hana", "erp"])

    if extras:
        consultas.append(f"{q} {' '.join(extras)}")

    return list(dict.fromkeys(consultas))


def _coletar_candidatos(db: FAISS, consultas: List[str], fetch_k: int) -> Tuple[List[Any], Dict[str, float]]:
    docs_unicos: Dict[str, Any] = {}
    melhor_distancia: Dict[str, float] = {}

    for consulta in consultas:
        docs_distancia = db.similarity_search_with_score(consulta, k=fetch_k)
        for doc, score in docs_distancia:
            chave = _chave_doc(doc)
            docs_unicos[chave] = doc
            if chave not in melhor_distancia:
                melhor_distancia[chave] = float(score)
            else:
                melhor_distancia[chave] = min(melhor_distancia[chave], float(score))

        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": fetch_k, "fetch_k": fetch_k, "lambda_mult": 0.35},
        )
        for doc in retriever.get_relevant_documents(consulta):
            chave = _chave_doc(doc)
            docs_unicos[chave] = doc

    return list(docs_unicos.values()), melhor_distancia


def _coletar_docs_do_docstore(db: FAISS) -> List[Any]:
    try:
        ids = list(db.index_to_docstore_id.values())
        return [db.docstore.search(doc_id) for doc_id in ids]
    except Exception:
        return []


def _coletar_candidatos_lexical(
    pergunta: str, docs: List[Any], fetch_k: int
) -> Tuple[List[Any], Dict[str, float]]:
    scored = []
    for doc in docs:
        score = _rerank_lexical(pergunta, [doc.page_content])[0]
        if score > 0:
            scored.append((doc, score))

    if not scored:
        return [], {}

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:fetch_k]
    # Distância sintética invertida para reaproveitar cálculo de confiança.
    dist_map = {_chave_doc(doc): (1.0 - float(score)) for doc, score in top}
    return [doc for doc, _ in top], dist_map


def _coletar_candidatos_lexical_multiconsulta(
    consultas: List[str], docs: List[Any], fetch_k: int
) -> Tuple[List[Any], Dict[str, float]]:
    melhor_score_por_doc: Dict[str, Tuple[Any, float]] = {}
    for consulta in consultas:
        cand_docs, _ = _coletar_candidatos_lexical(consulta, docs, fetch_k=fetch_k)
        for d in cand_docs:
            chave = _chave_doc(d)
            score = _rerank_lexical(consulta, [d.page_content])[0]
            atual = melhor_score_por_doc.get(chave)
            if atual is None or score > atual[1]:
                melhor_score_por_doc[chave] = (d, score)

    if not melhor_score_por_doc:
        return [], {}

    ordenado = sorted(
        melhor_score_por_doc.values(), key=lambda x: x[1], reverse=True
    )[:fetch_k]
    dist_map = {_chave_doc(doc): (1.0 - float(score)) for doc, score in ordenado}
    return [doc for doc, _ in ordenado], dist_map


def _filtrar_docs_por_escopo(docs: List[Any], escopo_sap: str | None) -> List[Any]:
    if not escopo_sap:
        return docs

    def _ok(source: str) -> bool:
        if escopo_sap == ESCOPO_ON_PREMISE:
            return "/SAP_S4HANA_ON-PREMISE/" in source
        if escopo_sap == ESCOPO_CLOUD:
            return "/SAP_S4HANA_CLOUD/" in source
        return True

    filtrados = []
    for d in docs:
        source = str(d.metadata.get("source", ""))
        if _ok(source):
            filtrados.append(d)
    return filtrados


def buscar_contexto(
    pergunta: str,
    k: int = 10,
    fetch_k: int = 80,
    escopo_sap: str | None = None,
) -> Dict[str, Any]:
    retorno_padrao = {
        "contexto": "",
        "fontes": [],
        "confianca": 0.0,
        "tem_contexto": False,
    }

    if not os.path.exists(VECTOR_PATH):
        return retorno_padrao

    try:
        embeddings = OpenAIEmbeddings()
        db = FAISS.load_local(
            VECTOR_PATH, embeddings, allow_dangerous_deserialization=True
        )
        consultas = _expandir_consultas(pergunta)
        try:
            docs_candidatos, mapa_distancias = _coletar_candidatos(
                db, consultas, fetch_k
            )
        except Exception:
            # Fallback sem rede: ranqueamento lexical sobre documentos já indexados.
            docs_locais = _coletar_docs_do_docstore(db)
            docs_candidatos, mapa_distancias = _coletar_candidatos_lexical_multiconsulta(
                consultas, docs_locais, fetch_k
            )

        docs_candidatos = _filtrar_docs_por_escopo(docs_candidatos, escopo_sap)
        if not docs_candidatos:
            return retorno_padrao

        textos = [doc.page_content for doc in docs_candidatos]
        scores_rerank = _rerank(pergunta, textos)
        docs_ordenados: List[Tuple[Any, float]] = sorted(
            zip(docs_candidatos, scores_rerank), key=lambda x: x[1], reverse=True
        )[:k]

        contexto_parts: List[str] = []
        fontes: List[str] = []
        confidencias: List[float] = []

        for idx, (doc, score_rerank) in enumerate(docs_ordenados, start=1):
            fonte = _normalizar_fonte(doc.metadata)
            fontes.append(fonte)
            score_dist = mapa_distancias.get(_chave_doc(doc), 2.0)
            conf_dist = _score_distancia_para_confianca(score_dist)
            conf_rerank = _score_rerank_para_confianca(score_rerank)
            conf = (0.55 * conf_dist) + (0.45 * conf_rerank)
            confidencias.append(conf)
            contexto_parts.append(
                f"[Trecho {idx} | Fonte: {fonte} | Relevancia_aprox: {conf:.3f}]\n"
                f"{doc.page_content}"
            )

        fontes_unicas = list(dict.fromkeys(fontes))
        confianca_media = sum(confidencias) / len(confidencias)

        return {
            "contexto": "\n\n".join(contexto_parts),
            "fontes": fontes_unicas,
            "confianca": round(confianca_media, 3),
            "tem_contexto": True,
        }
    except Exception:
        return retorno_padrao
