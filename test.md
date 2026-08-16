rag_app/
├── pyproject.toml
├── .env.example
├── src/
│   └── rag_app/
│       ├── config.py                    # pydantic-settings: wczytywanie .env
│       ├── domain/
│       │   ├── models.py                # CollectionConfig, FileRecord, Chunk, SearchResult...
│       │   ├── ingestion/
│       │   │   ├── scanner.py           # skan folderu, hash plików, wykrywanie nowych/zmienionych
│       │   │   ├── extractor.py         # docling — ekstrakcja tekstu/struktury z PDF
│       │   │   ├── chunker.py           # podział na chunki
│       │   │   └── page_renderer.py     # render stron PDF do obrazów (dla kolekcji wizualnej)
│       │   ├── embeddings/
│       │   │   ├── base.py              # interfejsy: DenseEmbedder, SparseEmbedder, VisualEmbedder
│       │   │   ├── dense_bge_m3.py
│       │   │   ├── sparse_splade.py
│       │   │   └── visual_colpali.py
│       │   ├── indexing/
│       │   │   ├── qdrant_client.py     # wrapper, tworzenie/kasowanie kolekcji
│       │   │   ├── text_indexer.py      # indeksowanie chunków (dense/sparse/hybrid)
│       │   │   └── visual_indexer.py    # indeksowanie stron jako multivector (ColPali)
│       │   ├── search/
│       │   │   ├── text_search.py
│       │   │   ├── visual_search.py
│       │   │   ├── fusion.py            # wrapper na ranx
│       │   │   └── reranking.py         # wrapper na reranker
│       │   ├── generation/
│       │   │   ├── base.py              # abstrakcyjny interfejs LLM
│       │   │   └── gemini.py
│       │   └── metadata/
│       │       └── store.py             # SQLite: rejestr kolekcji i plików
│       ├── services/
│       │   ├── collection_service.py    # orkiestracja: tworzenie/edycja kolekcji
│       │   └── query_service.py         # orkiestracja: search → fusion → rerank → generation
│       └── ui/
│           ├── main.py                  # punkt wejścia NiceGUI
│           ├── pages/
│           │   ├── collections_list.py
│           │   ├── collection_detail.py
│           │   ├── search_page.py
│           │   └── settings_page.py
│           └── components/
│               ├── progress_dialog.py
│               ├── file_table.py
│               └── result_card.py
└── tests/