from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None
HTTPX_AVAILABLE = importlib.util.find_spec("httpx") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE and HTTPX_AVAILABLE, "FastAPI/httpx are not installed")
class ApiTests(unittest.TestCase):
    def test_health_and_ask(self) -> None:
        from fastapi.testclient import TestClient

        from multimodal_rag.api import create_app
        from multimodal_rag.config import Settings
        from multimodal_rag.service import build_service

        settings = Settings(
            data_dir=ROOT / "data" / "sample_docs",
            embedder="hashing",
            reranker="lexical",
        )
        client = TestClient(create_app(build_service(settings)))

        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        response = client.post(
            "/ask",
            json={"question": "How does the platform combine vector and keyword search?", "top_k": 3},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["citations"])
        self.assertIn("answer", payload)


if __name__ == "__main__":
    unittest.main()
