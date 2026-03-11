import os
os.environ["USER_AGENT"] = "IAsmin/1.0"

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader

# =========================
# LISTA DE SITES
# =========================

lista_urls = [
    "https://help.sap.com/docs/SAP_S4HANA_ON-PREMISE/8308e6d301d54584a33cd04a9861bc52/2c0e7c571fbeb576e10000000a4450e5.html?locale=pt-BR&version=LATEST",
    "https://help.sap.com/docs/SAP_S4HANA_CLOUD/9d794cbd48c648bc8a176e422772de7e/7af7b8541486ed05e10000000a4450e5.html?locale=pt-BR&version=LATEST",
]

# =========================
# FUNÇÕES
# =========================

def carrega_sites(lista_urls):
    documentos = []

    for url in lista_urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            documentos.extend(docs)
            print(f"✅ Carregado: {url}")
        except Exception as e:
            print(f"❌ Erro em {url}: {e}")

    texto = "\n\n".join([doc.page_content for doc in documentos])
    return texto


def carrega_pdf(caminho):
    loader = PyPDFLoader(caminho)
    lista_documentos = loader.load()
    documento = "\n\n".join([doc.page_content for doc in lista_documentos])
    return documento
