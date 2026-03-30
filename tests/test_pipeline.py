from __future__ import annotations

from pathlib import Path
import unittest

from src.pipeline import run_pipeline
from src.sample_data import ensure_demo_dataset


class PipelineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = Path(__file__).resolve().parents[1]
        ensure_demo_dataset(self.base_dir)

    def test_pipeline_generates_results(self) -> None:
        report = run_pipeline(self.base_dir)
        self.assertGreaterEqual(report["catalog_size"], 6)
        self.assertEqual(report["top_match_id"], "VC-1001")
        self.assertGreater(report["top_match_score"], 0.25)
        self.assertEqual(len(report["results"]), 3)


if __name__ == "__main__":
    unittest.main()

