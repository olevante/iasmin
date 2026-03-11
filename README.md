# IAsmin – Consultora SAP (RAG + Agente)

Assistente SAP com RAG, cache e busca web opcional. Suporta escopo por versão (On-Premise vs Cloud Public) e geração de documentos (BBP). Otimizado para reduzir custo de LLM via cache persistente.

## Stack
- Python 3.11/3.12
- Streamlit
- LangChain + FAISS
- Playwright (crawler)
- SearxNG (busca web open-source, opcional)
- YouTube Transcript API (transcrições quando disponíveis)

## Estrutura
- `app.py` — Interface Streamlit + orquestração.
- `rag/ingestao.py` — Crawler + indexação (full/incremental).
- `rag/busca.py` — Recuperação e filtro por escopo SAP.
- `rag/qa_cache.py` — Cache persistente de respostas (SQLite).
- `rag/web_search.py` — Busca web via SearxNG (HTML fallback).
- `rag/web_fetch.py` — Fetch de páginas web.
- `rag/youtube_transcript.py` — Transcrição YouTube.
- `templates/bbp_template.md` — Template BBP.

## Pré‑requisitos
- Python 3.11/3.12
- Docker (opcional, para SearxNG)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Crie `.env` (ou copie de `.env.example`):
```env
OPENAI_API_KEY=seu_token
```

## Rodar o app
```bash
source .venv/bin/activate
python -m streamlit run app.py
```

## Deploy (Railway)
1. Faça push do repositório para o GitHub.
2. Crie um novo projeto no Railway e conecte ao repo.
3. Configure as variáveis de ambiente:
   - `OPENAI_API_KEY`
   - `SEARXNG_URL` (opcional)
   - `IASMIN_DATA_DIR=/data/iasmin` (recomendado com volume)
   - `IASMIN_CACHE_DIR=/data/iasmin/cache` (opcional)
4. Anexe um volume persistente e aponte para `/data`.
4. O start command já está definido em `railway.json`.
5. Se precisar de Playwright, configure o Railway para usar o `Dockerfile` (Build → Dockerfile).

## Crawler e indexação
### Full rebuild (do zero)
```bash
python3 -m rag.reindexar_sap --forcar-redescoberta --crawler hybrid --max-paginas 10000 --profundidade-max 30 --max-iteracoes 80 --full-rebuild
```

### Incremental (delta)
```bash
python3 -m rag.reindexar_sap --forcar-redescoberta --crawler hybrid --continuar-catalogo --max-paginas 10000 --profundidade-max 30 --max-iteracoes 80
```

## SearxNG (busca web open‑source)
Opcional. Se tiver SearxNG rodando:
```bash
export SEARXNG_URL="http://localhost:8080/search"
```

A integração usa fallback HTML se a API JSON bloquear.

### Deploy SearxNG no Railway (serviço separado)
- Crie um novo serviço Docker com imagem `searxng/searxng`.
- Exponha porta 8080 e crie variável `SEARXNG_URL` no app principal apontando para `https://<seu-searxng>/search`.

## YouTube
Se houver transcrição pública, ela será usada automaticamente.

## Comandos no chat
- `bbp` / `gerar bbp` / `/bbp` — mostra template BBP.
- `trocar versão sap` — reseta escopo.

## Escopo SAP
Na primeira pergunta, o app solicita a versão:
- On-Premise / ECC / EHP / RISE → usa apenas fontes `SAP_S4HANA_ON-PREMISE`
- Public / GROW → usa apenas fontes `SAP_S4HANA_CLOUD`

## Assets
Coloque seus logos em:
- `assets/logo_nova.png`
- `assets/iasmin.png`

## Licenças e fontes
Este projeto consulta fontes públicas. Ajuste conforme as políticas de uso do seu ambiente.
