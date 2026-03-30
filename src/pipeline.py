from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.retrieval import load_engine, results_to_frame
from src.sample_data import ensure_demo_dataset


DEFAULT_QUERY = {
    "query_text": "smartphone com tela quebrada e rachadura visível na parte superior",
    "query_image_file": "cracked_screen_phone.png",
}


def run_pipeline(base_dir: str | Path) -> dict[str, Any]:
    base_path = Path(base_dir)
    ensure_demo_dataset(base_path)
    engine = load_engine(base_path)
    query_image_path = base_path / "data" / "raw" / "images" / DEFAULT_QUERY["query_image_file"]
    results = engine.search(DEFAULT_QUERY["query_text"], str(query_image_path), top_k=3)
    results_frame = results_to_frame(results)

    report = {
        "runtime_mode": engine.mode,
        "query_text": DEFAULT_QUERY["query_text"],
        "query_image_path": str(query_image_path.resolve()),
        "catalog_size": int(len(engine.dataframe)),
        "top_match_id": results[0].complaint_id,
        "top_match_score": round(results[0].similarity_score, 4),
        "results": results_frame.to_dict(orient="records"),
    }

    processed_dir = base_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "retrieval_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report

