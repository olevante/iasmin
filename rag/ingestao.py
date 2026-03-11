import os
from collections import deque
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import re
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR_ENV = os.getenv("IASMIN_DATA_DIR", "").strip()
DATA_DIR = Path(DATA_DIR_ENV).expanduser() if DATA_DIR_ENV else ROOT_DIR

(DATA_DIR / "rag").mkdir(parents=True, exist_ok=True)


# =========================
# CONFIGURAÇÕES
# =========================

PASTA_PDFS = "arquivos"
ARQUIVO_FAISS = str(DATA_DIR / "rag" / "base_vetorial")
URLS_SALVAS = str(DATA_DIR / "rag" / "sap_urls.txt")
ARQUIVO_CATALOGO = str(DATA_DIR / "rag" / "sap_url_catalog.json")
ARQUIVO_HASH_CACHE = str(DATA_DIR / "rag" / "sap_url_hash_cache.json")
DOMINIO_SAP_HELP = "help.sap.com"
USER_AGENT = "IAsmin-SAP-Crawler/1.0"


# =========================
# LISTA DE URLS INICIAIS
# =========================

SEMENTES_SAP = [
    "https://help.sap.com/docs/SAP_S4HANA_ON-PREMISE/8308e6d301d54584a33cd04a9861bc52/2c0e7c571fbeb576e10000000a4450e5.html?locale=pt-BR&version=LATEST",
    "https://help.sap.com/docs/SAP_S4HANA_CLOUD/9d794cbd48c648bc8a176e422772de7e/7af7b8541486ed05e10000000a4450e5.html?locale=pt-BR&version=LATEST",
]


# =========================
# DESCOBERTA SAP HELP
# =========================

def _normalizar_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query_limpa = {}
    if "locale" in query:
        query_limpa["locale"] = query["locale"][0]
    if "version" in query:
        query_limpa["version"] = query["version"][0]
    query_str = urlencode(query_limpa)
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, query_str, "")
    )


def _url_sap_valida(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.netloc.endswith(DOMINIO_SAP_HELP)
        and "/docs/" in parsed.path
    )


def _extrair_escopos_guias(sementes: list[str]) -> list[str]:
    escopos = set()
    for url in sementes:
        parsed = urlparse(url)
        partes = [p for p in parsed.path.split("/") if p]
        # Esperado: /docs/{produto}/{guia}/{topico}.html
        if len(partes) >= 4 and partes[0] == "docs":
            escopos.add(f"/docs/{partes[1]}/{partes[2]}/")
    return sorted(escopos)


def _url_esta_no_escopo(url: str, escopos: list[str]) -> bool:
    if not escopos:
        return True
    parsed = urlparse(url)
    return any(parsed.path.startswith(escopo) for escopo in escopos)


def _extrair_urls_do_texto(base_url: str, texto: str) -> set[str]:
    candidatas = set()
    if not texto:
        return candidatas

    texto = texto.replace("\\/", "/")

    for match in re.findall(
        r'https?://help\.sap\.com/docs/[^"\'>\s]+?\.html(?:\?[^"\'>\s]*)?', texto
    ):
        candidatas.add(match)

    for match in re.findall(r'/docs/[^"\'>\s]+?\.html(?:\?[^"\'>\s]*)?', texto):
        candidatas.add(urljoin(base_url, match))

    return candidatas


def _extrair_candidatas(base_url: str, html: str, soup: BeautifulSoup) -> list[str]:
    candidatas = set()

    for anchor in soup.find_all("a", href=True):
        candidatas.add(urljoin(base_url, anchor["href"]))

    # Alguns links do SAP Help ficam em atributos não padronizados.
    for tag in soup.find_all(True):
        for valor in tag.attrs.values():
            valores = valor if isinstance(valor, list) else [valor]
            for item in valores:
                if isinstance(item, str):
                    candidatas.update(_extrair_urls_do_texto(base_url, item))

    candidatas.update(_extrair_urls_do_texto(base_url, html))
    return [_normalizar_url(url) for url in candidatas]


def descobrir_urls_sap(
    sementes: list[str], max_paginas: int = 120, profundidade_max: int = 2
) -> list[str]:
    visitadas = set()
    escopos = _extrair_escopos_guias(sementes)
    fila = deque((url, 0) for url in sementes if _url_sap_valida(url))
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    while fila and len(visitadas) < max_paginas:
        atual, profundidade = fila.popleft()
        atual = _normalizar_url(atual)

        if atual in visitadas:
            continue
        if not _url_esta_no_escopo(atual, escopos):
            continue
        visitadas.add(atual)

        if profundidade >= profundidade_max:
            continue

        try:
            resposta = session.get(atual, timeout=20)
            if resposta.status_code >= 400:
                continue
            soup = BeautifulSoup(resposta.text, "html.parser")
        except Exception:
            continue

        candidatas = _extrair_candidatas(atual, resposta.text, soup)

        for candidata in candidatas:
            if len(visitadas) + len(fila) >= max_paginas:
                break
            if (
                _url_sap_valida(candidata)
                and _url_esta_no_escopo(candidata, escopos)
                and candidata not in visitadas
            ):
                fila.append((candidata, profundidade + 1))

    return sorted(visitadas)


def salvar_urls(urls: list[str], caminho: str = URLS_SALVAS) -> None:
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(urls))


def carregar_urls(caminho: str = URLS_SALVAS) -> list[str]:
    if not os.path.exists(caminho):
        return []
    with open(caminho, "r", encoding="utf-8") as f:
        return [linha.strip() for linha in f if linha.strip()]


def carregar_hash_cache(caminho: str = ARQUIVO_HASH_CACHE) -> dict:
    if not os.path.exists(caminho):
        return {}
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def salvar_hash_cache(cache: dict, caminho: str = ARQUIVO_HASH_CACHE) -> None:
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _catalogo_vazio() -> dict:
    return {"visited": [], "pending": [], "depth_map": {}}


def carregar_catalogo(caminho: str = ARQUIVO_CATALOGO) -> dict:
    if not os.path.exists(caminho):
        return _catalogo_vazio()
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "visited": data.get("visited", []),
            "pending": data.get("pending", []),
            "depth_map": data.get("depth_map", {}),
        }
    except Exception:
        return _catalogo_vazio()


def salvar_catalogo(
    visited: set[str],
    pending: deque[tuple[str, int]],
    depth_map: dict[str, int],
    caminho: str = ARQUIVO_CATALOGO,
) -> None:
    data = {
        "visited": sorted(visited),
        "pending": list(pending),
        "depth_map": depth_map,
    }
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def descobrir_urls_sap_recursivo(
    sementes: list[str],
    max_paginas: int = 5000,
    max_profundidade: int = 20,
    continuar_catalogo: bool = True,
    bootstrap_urls: list[str] | None = None,
    salvar_cada: int = 50,
) -> list[str]:
    escopos = _extrair_escopos_guias(sementes)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    if continuar_catalogo:
        cat = carregar_catalogo()
        visitadas = set(cat["visited"])
        fila = deque((u, int(d)) for u, d in cat["pending"])
        depth_map = {k: int(v) for k, v in cat["depth_map"].items()}
    else:
        visitadas = set()
        fila = deque()
        depth_map = {}

    candidatas_iniciais = list(sementes)
    if bootstrap_urls:
        candidatas_iniciais.extend(bootstrap_urls)

    for url in candidatas_iniciais:
        norm = _normalizar_url(url)
        if (
            _url_sap_valida(norm)
            and _url_esta_no_escopo(norm, escopos)
            and norm not in depth_map
        ):
            depth_map[norm] = 0
            fila.append((norm, 0))

    processadas = 0
    while fila and len(visitadas) < max_paginas:
        atual, profundidade = fila.popleft()
        atual = _normalizar_url(atual)

        if atual in visitadas:
            continue
        if profundidade > max_profundidade:
            continue
        if not _url_sap_valida(atual) or not _url_esta_no_escopo(atual, escopos):
            continue

        try:
            resposta = session.get(atual, timeout=20)
            if resposta.status_code >= 400:
                continue
            soup = BeautifulSoup(resposta.text, "html.parser")
        except Exception:
            continue

        visitadas.add(atual)
        processadas += 1

        for candidata in _extrair_candidatas(atual, resposta.text, soup):
            if (
                _url_sap_valida(candidata)
                and _url_esta_no_escopo(candidata, escopos)
                and candidata not in visitadas
            ):
                prox_depth = profundidade + 1
                if prox_depth <= max_profundidade:
                    old = depth_map.get(candidata)
                    if old is None or prox_depth < old:
                        depth_map[candidata] = prox_depth
                        fila.append((candidata, prox_depth))

        if processadas % salvar_cada == 0:
            salvar_catalogo(visitadas, fila, depth_map)

    salvar_catalogo(visitadas, fila, depth_map)
    return sorted(visitadas)


# =========================
# CARREGAR DOCUMENTOS
# =========================

def carregar_docs(urls: list[str]):
    docs = []

    # ---------- PDFs ----------
    if os.path.isdir(PASTA_PDFS):
        for arquivo in os.listdir(PASTA_PDFS):
            if arquivo.endswith(".pdf"):
                caminho = os.path.join(PASTA_PDFS, arquivo)
                loader = PyPDFLoader(caminho)
                docs.extend(loader.load())

    # ---------- URLs ----------
    docs.extend(carregar_docs_urls(urls))

    return docs


def carregar_docs_urls(urls: list[str]) -> list[Document]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    docs = []

    for url in urls:
        try:
            resp = session.get(url, timeout=25)
            if resp.status_code >= 400:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            texto = soup.get_text(separator="\n", strip=True)
            if not texto:
                continue
            docs.append(
                Document(
                    page_content=texto,
                    metadata={"source": _normalizar_url(url)},
                )
            )
        except Exception:
            continue

    return docs


def _hash_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8", errors="ignore")).hexdigest()


def _agora_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coletar_docs_urls_com_hash(urls: list[str]) -> tuple[list[Document], dict, list[str]]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    docs = []
    hash_map = {}
    falhas = []

    for url in urls:
        url_norm = _normalizar_url(url)
        try:
            resp = session.get(url_norm, timeout=25)
            if resp.status_code >= 400:
                falhas.append(url_norm)
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            texto = soup.get_text(separator="\n", strip=True)
            if not texto:
                falhas.append(url_norm)
                continue
            h = _hash_texto(texto)
            hash_map[url_norm] = {
                "hash": h,
                "status_code": resp.status_code,
                "fetched_at": _agora_iso(),
            }
            docs.append(Document(page_content=texto, metadata={"source": url_norm}))
        except Exception:
            falhas.append(url_norm)

    return docs, hash_map, falhas


def _obter_ids_por_source(db: FAISS, sources: set[str]) -> list[str]:
    ids = []
    for doc_id in db.index_to_docstore_id.values():
        doc = db.docstore.search(doc_id)
        if doc and doc.metadata.get("source") in sources:
            ids.append(doc_id)
    return ids


def _aplicar_incremental_faiss(
    urls: list[str],
    splitter: RecursiveCharacterTextSplitter,
    embeddings: OpenAIEmbeddings,
) -> None:
    cache_antigo = carregar_hash_cache()
    docs_atuais, hash_atual, falhas = _coletar_docs_urls_com_hash(urls)
    docs_por_url = {d.metadata["source"]: d for d in docs_atuais}

    urls_atuais = set(hash_atual.keys())
    urls_anteriores = set(cache_antigo.keys())
    removidas = urls_anteriores - urls_atuais

    novas_ou_alteradas = set()
    for u in urls_atuais:
        h_antigo = cache_antigo.get(u, {}).get("hash")
        h_novo = hash_atual.get(u, {}).get("hash")
        if h_antigo != h_novo:
            novas_ou_alteradas.add(u)

    path_faiss = Path(ARQUIVO_FAISS)
    existe_base = (path_faiss / "index.faiss").exists() and (path_faiss / "index.pkl").exists()

    if existe_base:
        db = FAISS.load_local(
            ARQUIVO_FAISS, embeddings, allow_dangerous_deserialization=True
        )
    else:
        db = None

    sources_para_remover = removidas.union(novas_ou_alteradas)
    if db and sources_para_remover:
        ids_para_remover = _obter_ids_por_source(db, sources_para_remover)
        if ids_para_remover:
            db.delete(ids=ids_para_remover)

    docs_delta = [docs_por_url[u] for u in novas_ou_alteradas if u in docs_por_url]

    docs_pdf = []
    if not existe_base and os.path.isdir(PASTA_PDFS):
        for arquivo in os.listdir(PASTA_PDFS):
            if arquivo.endswith(".pdf"):
                caminho = os.path.join(PASTA_PDFS, arquivo)
                docs_pdf.extend(PyPDFLoader(caminho).load())

    docs_para_indexar = docs_pdf + docs_delta
    if docs_para_indexar:
        chunks = splitter.split_documents(docs_para_indexar)
        if db is None:
            db = FAISS.from_documents(chunks, embeddings)
        else:
            db.add_documents(chunks)

    if db is None:
        # Se nada para indexar e não há base, cria base mínima vazia de forma explícita.
        raise RuntimeError(
            "Não foi possível criar base incremental: nenhum documento válido foi carregado."
        )

    db.save_local(ARQUIVO_FAISS)

    # Mantém no cache apenas URLs válidas da lista final.
    cache_novo = {u: hash_atual[u] for u in urls if u in hash_atual}
    salvar_hash_cache(cache_novo)

    print(
        "Incremental concluído | "
        f"novas/alteradas={len(novas_ou_alteradas)} "
        f"removidas={len(removidas)} "
        f"falhas_download={len(falhas)}"
    )


# =========================
# CRIAR BASE VETORIAL
# =========================

def criar_base(
    max_paginas: int = 5000,
    profundidade_max: int = 20,
    forcar_redescoberta: bool = False,
    modo_incremental: bool = True,
):
    if forcar_redescoberta or not os.path.exists(URLS_SALVAS):
        print("Descobrindo páginas SAP Help...")
        urls = descobrir_urls_sap_recursivo(
            sementes=SEMENTES_SAP,
            max_paginas=max_paginas,
            max_profundidade=profundidade_max,
            continuar_catalogo=False,
        )
        salvar_urls(urls)
    else:
        urls = carregar_urls()

    print(f"URLs SAP para indexação: {len(urls)}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = OpenAIEmbeddings()

    if modo_incremental:
        print("Aplicando indexação incremental...")
        _aplicar_incremental_faiss(urls, splitter, embeddings)
        print("✅ Base incremental atualizada com sucesso!")
        return

    print("Carregando documentos para rebuild completo...")
    docs = carregar_docs(urls)
    print("Dividindo em chunks...")
    chunks = splitter.split_documents(docs)
    print("Criando embeddings e reconstruindo FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Salvando base vetorial...")
    vectorstore.save_local(ARQUIVO_FAISS)
    cache_pos_rebuild = {}
    for d in docs:
        source = d.metadata.get("source", "")
        if isinstance(source, str) and source.startswith("http"):
            source_norm = _normalizar_url(source)
            cache_pos_rebuild[source_norm] = {
                "hash": _hash_texto(d.page_content),
                "status_code": 200,
                "fetched_at": _agora_iso(),
            }
    salvar_hash_cache(cache_pos_rebuild)
    print("✅ Base recriada com sucesso!")


if __name__ == "__main__":
    criar_base()
