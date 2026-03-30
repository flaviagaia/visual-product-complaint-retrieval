from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def extract_visual_features(image_path: str | Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((96, 96))
    histogram = np.asarray(image.histogram(), dtype=float)
    histogram = histogram / max(histogram.sum(), 1.0)

    grayscale = image.convert("L")
    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    edge_array = np.asarray(edges, dtype=float) / 255.0

    features = np.concatenate(
        [
            histogram[:96],
            np.array(
                [
                    edge_array.mean(),
                    edge_array.std(),
                    float(np.percentile(edge_array, 90)),
                ]
            ),
        ]
    )
    return _normalize(features)


@dataclass
class RetrievalResult:
    complaint_id: str
    title: str
    product_category: str
    defect_type: str
    expected_resolution: str
    similarity_score: float
    text_similarity: float
    image_similarity: float
    metadata_match: float
    explanation: str
    image_path: str
    complaint_text: str


class GeminiEmbeddingClient:
    def __init__(self) -> None:
        self.model_name = "gemini-embedding-2-preview"
        self._client = None
        self._types = None
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return
        try:
            from google import genai
            from google.genai import types
        except Exception:
            return

        self._client = genai.Client(api_key=api_key)
        self._types = types

    @property
    def enabled(self) -> bool:
        return self._client is not None and self._types is not None

    def embed_text(self, text: str) -> np.ndarray:
        response = self._client.models.embed_content(
            model=self.model_name,
            contents=text,
        )
        return np.asarray(response.embeddings[0].values, dtype=float)

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "image/png"
        image_bytes = Path(image_path).read_bytes()
        part = self._types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        response = self._client.models.embed_content(
            model=self.model_name,
            contents=[part],
        )
        return np.asarray(response.embeddings[0].values, dtype=float)


class ComplaintRetrievalEngine:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe.copy()
        self.vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 2))
        corpus = (
            self.dataframe["title"].fillna("")
            + " "
            + self.dataframe["complaint_text"].fillna("")
            + " "
            + self.dataframe["product_category"].fillna("")
            + " "
            + self.dataframe["defect_type"].fillna("")
        ).tolist()
        self.text_matrix = self.vectorizer.fit_transform(corpus)
        self.visual_matrix = np.vstack(
            [extract_visual_features(path) for path in self.dataframe["image_path"].tolist()]
        )
        self.gemini = GeminiEmbeddingClient()
        if self.gemini.enabled:
            self.mode = "gemini_embedding_2"
            self._catalog_embeddings = self._build_gemini_catalog_embeddings()
        else:
            self.mode = "local_multimodal_fallback"
            self._catalog_embeddings = None

    def _build_gemini_catalog_embeddings(self) -> np.ndarray:
        catalog_embeddings = []
        for _, row in self.dataframe.iterrows():
            text_embedding = self.gemini.embed_text(
                f"{row['title']} {row['complaint_text']} {row['product_category']} {row['defect_type']}"
            )
            image_embedding = self.gemini.embed_image(row["image_path"])
            catalog_embeddings.append(_normalize(0.6 * _normalize(text_embedding) + 0.4 * _normalize(image_embedding)))
        return np.vstack(catalog_embeddings)

    def _build_local_query_vector(self, query_text: str, query_image_path: str | None) -> tuple[np.ndarray, np.ndarray]:
        text_vector = self.vectorizer.transform([query_text]).toarray()[0]
        text_vector = _normalize(text_vector)
        if query_image_path:
            image_vector = extract_visual_features(query_image_path)
        else:
            image_vector = np.zeros(self.visual_matrix.shape[1], dtype=float)
        return text_vector, image_vector

    def _build_gemini_query_vector(self, query_text: str, query_image_path: str | None) -> np.ndarray:
        text_embedding = self.gemini.embed_text(query_text)
        if query_image_path:
            image_embedding = self.gemini.embed_image(query_image_path)
            return _normalize(0.6 * _normalize(text_embedding) + 0.4 * _normalize(image_embedding))
        return _normalize(text_embedding)

    def search(
        self,
        query_text: str,
        query_image_path: str | None = None,
        top_k: int = 3,
    ) -> list[RetrievalResult]:
        if self.mode == "gemini_embedding_2":
            query_vector = self._build_gemini_query_vector(query_text, query_image_path)
            similarities = cosine_similarity([query_vector], self._catalog_embeddings)[0]
            text_similarities = np.zeros_like(similarities)
            image_similarities = np.zeros_like(similarities)
        else:
            text_vector, image_vector = self._build_local_query_vector(query_text, query_image_path)
            text_similarities = cosine_similarity([text_vector], self.text_matrix)[0]
            if np.allclose(image_vector, 0):
                image_similarities = np.zeros(len(self.dataframe))
            else:
                image_similarities = cosine_similarity([image_vector], self.visual_matrix)[0]
            similarities = 0.7 * text_similarities + 0.3 * image_similarities

        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        results: list[RetrievalResult] = []
        query_lower = query_text.lower()
        for idx in ranked_indices:
            row = self.dataframe.iloc[idx]
            metadata_match = 0.0
            if row["product_category"] in query_lower:
                metadata_match += 0.5
            if row["defect_type"].replace("_", " ") in query_lower:
                metadata_match += 0.5

            explanation_parts = []
            if text_similarities[idx] > 0:
                explanation_parts.append(f"texto={text_similarities[idx]:.2f}")
            if image_similarities[idx] > 0:
                explanation_parts.append(f"imagem={image_similarities[idx]:.2f}")
            if metadata_match > 0:
                explanation_parts.append(f"metadados={metadata_match:.2f}")
            if not explanation_parts and self.mode == "gemini_embedding_2":
                explanation_parts.append("match multimodal calculado via Gemini Embedding 2")

            results.append(
                RetrievalResult(
                    complaint_id=row["complaint_id"],
                    title=row["title"],
                    product_category=row["product_category"],
                    defect_type=row["defect_type"],
                    expected_resolution=row["expected_resolution"],
                    similarity_score=float(similarities[idx] + 0.05 * metadata_match),
                    text_similarity=float(text_similarities[idx]),
                    image_similarity=float(image_similarities[idx]),
                    metadata_match=float(metadata_match),
                    explanation=" | ".join(explanation_parts),
                    image_path=row["image_path"],
                    complaint_text=row["complaint_text"],
                )
            )
        return results


def load_engine(base_dir: str | Path) -> ComplaintRetrievalEngine:
    csv_path = Path(base_dir) / "data" / "raw" / "complaints_catalog.csv"
    dataframe = pd.read_csv(csv_path)
    return ComplaintRetrievalEngine(dataframe)


def results_to_frame(results: list[RetrievalResult]) -> pd.DataFrame:
    return pd.DataFrame([result.__dict__ for result in results])

