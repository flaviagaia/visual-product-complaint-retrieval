from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipeline import DEFAULT_QUERY
from src.retrieval import load_engine, results_to_frame
from src.sample_data import ensure_demo_dataset


BASE_DIR = Path(__file__).resolve().parent


def _save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def main() -> None:
    st.set_page_config(page_title="Visual Product Complaint Retrieval", layout="wide")
    ensure_demo_dataset(BASE_DIR)
    engine = load_engine(BASE_DIR)

    st.title("Visual Product Complaint Retrieval")
    st.caption(
        "MVP de retrieval multimodal para reclamações visuais de produto com rota opcional para Gemini Embedding 2 "
        "e fallback local com TF-IDF + features visuais."
    )

    st.sidebar.header("Configuração da Busca")
    query_text = st.sidebar.text_area(
        "Descrição da reclamação",
        value=DEFAULT_QUERY["query_text"],
        height=140,
    )
    uploaded_image = st.sidebar.file_uploader(
        "Imagem da reclamação",
        type=["png", "jpg", "jpeg"],
    )
    top_k = st.sidebar.slider("Top-K", min_value=1, max_value=5, value=3)

    if uploaded_image is not None:
        query_image_path = _save_uploaded_file(uploaded_image)
    else:
        query_image_path = str(
            (BASE_DIR / "data" / "raw" / "images" / DEFAULT_QUERY["query_image_file"]).resolve()
        )

    left, right = st.columns([1.1, 1.6])

    with left:
        st.subheader("Consulta")
        st.image(query_image_path, caption="Imagem usada na busca", use_container_width=True)
        st.metric("Modo de execução", engine.mode)
        st.metric("Itens no catálogo", len(engine.dataframe))

        st.subheader("Stack técnica")
        st.markdown(
            "- `Gemini Embedding 2` como rota multimodal opcional\n"
            "- `TF-IDF + cosine similarity` para retrieval textual local\n"
            "- `Pillow` para extração de sinais visuais\n"
            "- `Streamlit` para inspeção e validação manual"
        )

    results = engine.search(query_text=query_text, query_image_path=query_image_path, top_k=top_k)
    results_frame = results_to_frame(results)

    with right:
        st.subheader("Resultados Recuperados")
        if results:
            st.metric("Melhor match", results[0].complaint_id)
            st.metric("Score do melhor match", f"{results[0].similarity_score:.3f}")

        for result in results:
            with st.container(border=True):
                cols = st.columns([1.1, 1.6])
                with cols[0]:
                    st.image(result.image_path, caption=result.complaint_id, use_container_width=True)
                with cols[1]:
                    st.markdown(f"**{result.title}**")
                    st.write(result.complaint_text)
                    st.markdown(
                        f"`categoria` {result.product_category} | "
                        f"`defeito` {result.defect_type} | "
                        f"`resolução esperada` {result.expected_resolution}"
                    )
                    st.progress(min(max(result.similarity_score, 0.0), 1.0))
                    st.caption(
                        f"score={result.similarity_score:.3f} | "
                        f"texto={result.text_similarity:.3f} | "
                        f"imagem={result.image_similarity:.3f}"
                    )
                    st.caption(f"explicação do match: {result.explanation}")

        st.subheader("Tabela técnica")
        st.dataframe(
            results_frame[
                [
                    "complaint_id",
                    "similarity_score",
                    "text_similarity",
                    "image_similarity",
                    "metadata_match",
                    "product_category",
                    "defect_type",
                ]
            ],
            use_container_width=True,
        )

    st.subheader("Catálogo de referência")
    st.dataframe(
        pd.read_csv(BASE_DIR / "data" / "raw" / "complaints_catalog.csv")[
            [
                "complaint_id",
                "product_category",
                "brand",
                "title",
                "defect_type",
                "severity",
                "expected_resolution",
            ]
        ],
        use_container_width=True,
    )


if __name__ == "__main__":
    main()

