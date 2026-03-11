import re
from typing import List
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse


DOMINIO_SAP_HELP = "help.sap.com"
USER_AGENT = "IAsmin-SAP-Crawler/1.0 (Playwright)"


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
        # /docs/{produto}/{guia}/{topico}.html
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


def descobrir_urls_sap_playwright(
    sementes: List[str],
    max_paginas: int = 3000,
    max_iteracoes: int = 40,
    espera_ms: int = 1200,
    headless: bool = True,
) -> List[str]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError(
            "Playwright não está disponível. Instale com `pip install playwright` "
            "e `playwright install chromium`."
        ) from exc

    escopos = _extrair_escopos_guias(sementes)
    urls = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()

        for seed in sementes:
            if len(urls) >= max_paginas:
                break
            if not _url_sap_valida(seed):
                continue

            try:
                page.goto(seed, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(1500)
            except Exception:
                continue

            for _ in range(max_iteracoes):
                if len(urls) >= max_paginas:
                    break

                # Captura links renderizados na página (inclui árvore lateral quando presente).
                hrefs = page.evaluate(
                    """
                    () => {
                      const out = [];
                      const els = document.querySelectorAll("a[href], [href], [data-href]");
                      for (const el of els) {
                        const vals = [el.getAttribute("href"), el.getAttribute("data-href")];
                        for (const v of vals) {
                          if (v && typeof v === "string") out.push(v);
                        }
                      }
                      // Inclui scripts inline para capturar rotas dinâmicas.
                      for (const s of document.querySelectorAll("script")) {
                        if (s.textContent) out.push(s.textContent);
                      }
                      return out;
                    }
                    """
                )

                antes = len(urls)
                for href in hrefs:
                    if not isinstance(href, str):
                        continue
                    for candidata in _extrair_urls_do_texto(page.url, href):
                        candidata = _normalizar_url(candidata)
                        if (
                            _url_sap_valida(candidata)
                            and _url_esta_no_escopo(candidata, escopos)
                        ):
                            urls.add(candidata)
                            if len(urls) >= max_paginas:
                                break
                    if len(urls) >= max_paginas:
                        break

                # Tenta expandir nós colapsados do menu/TOC.
                clicados = page.evaluate(
                    """
                    () => {
                      let clicks = 0;
                      const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                      const candidates = new Set();

                      for (const el of document.querySelectorAll('[aria-expanded="false"]')) candidates.add(el);
                      for (const el of document.querySelectorAll('button, [role="button"], [class*="toggle" i], [class*="expand" i], [class*="tree" i]')) candidates.add(el);

                      for (const el of candidates) {
                        if (!isVisible(el)) continue;
                        const t = ((el.innerText || '') + ' ' + (el.getAttribute('aria-label') || '') + ' ' + (el.getAttribute('title') || '')).toLowerCase();
                        const expandable = el.getAttribute('aria-expanded') === 'false' || t.includes('expand') || t.includes('mais') || t.includes('more');
                        if (!expandable) continue;
                        try { el.click(); clicks += 1; } catch (_) {}
                        if (clicks >= 80) break;
                      }
                      return clicks;
                    }
                    """
                )

                # Scroll ajuda a carregar itens lazy da árvore.
                page.mouse.wheel(0, 6000)
                page.wait_for_timeout(200)
                page.mouse.wheel(0, -3000)
                page.wait_for_timeout(espera_ms)

                sem_novos = len(urls) == antes
                if sem_novos and int(clicados) == 0:
                    break

        context.close()
        browser.close()

    # Garante inclusão explícita das seeds.
    for seed in sementes:
        if _url_sap_valida(seed):
            urls.add(_normalizar_url(seed))

    return sorted(urls)[:max_paginas]
