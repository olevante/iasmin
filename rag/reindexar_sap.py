import argparse

from rag.ingestao import (
    SEMENTES_SAP,
    carregar_urls,
    criar_base,
    descobrir_urls_sap,
    descobrir_urls_sap_recursivo,
    salvar_urls,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descobre URLs SAP Help e recria a base vetorial."
    )
    parser.add_argument(
        "--max-paginas",
        type=int,
        default=5000,
        help="Máximo de páginas SAP Help para descoberta/indexação.",
    )
    parser.add_argument(
        "--profundidade-max",
        type=int,
        default=20,
        help="Profundidade de navegação a partir das sementes.",
    )
    parser.add_argument(
        "--forcar-redescoberta",
        action="store_true",
        help="Ignora o cache de URLs salvo e faz nova descoberta.",
    )
    parser.add_argument(
        "--crawler",
        choices=["requests", "playwright", "hybrid"],
        default="hybrid",
        help="Estratégia de descoberta de URLs filhas do SAP Help.",
    )
    parser.add_argument(
        "--max-iteracoes",
        type=int,
        default=40,
        help="Apenas no modo playwright: iterações de expansão por URL pai.",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Apenas no modo playwright: abre navegador visível para depuração.",
    )
    parser.add_argument(
        "--continuar-catalogo",
        action="store_true",
        help="Continua a descoberta recursiva do último estado salvo.",
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Desativa incremental e recria a base vetorial completa.",
    )
    args = parser.parse_args()

    if args.forcar_redescoberta:
        bootstrap_urls = []
        if args.crawler in {"playwright", "hybrid"}:
            from rag.crawler_playwright import descobrir_urls_sap_playwright

            print("Descobrindo URLs SAP com Playwright...")
            bootstrap_urls = descobrir_urls_sap_playwright(
                sementes=SEMENTES_SAP,
                max_paginas=args.max_paginas,
                max_iteracoes=args.max_iteracoes,
                headless=not args.headful,
            )
            print(f"Bootstrap Playwright: {len(bootstrap_urls)} URLs")

        if args.crawler == "requests":
            print("Descobrindo URLs SAP com requests/BS4...")
            urls = descobrir_urls_sap(
                sementes=SEMENTES_SAP,
                max_paginas=args.max_paginas,
                profundidade_max=args.profundidade_max,
            )
        else:
            print("Expandindo recursivamente em todas as sub-URLs...")
            urls = descobrir_urls_sap_recursivo(
                sementes=SEMENTES_SAP,
                max_paginas=args.max_paginas,
                max_profundidade=args.profundidade_max,
                continuar_catalogo=args.continuar_catalogo,
                bootstrap_urls=bootstrap_urls,
            )

        salvar_urls(urls)
        print(f"URLs descobertas: {len(urls)}")
    else:
        urls = carregar_urls()
        print(f"Usando URLs já salvas: {len(urls)}")

    criar_base(
        max_paginas=args.max_paginas,
        profundidade_max=args.profundidade_max,
        forcar_redescoberta=False,
        modo_incremental=not args.full_rebuild,
    )


if __name__ == "__main__":
    main()
