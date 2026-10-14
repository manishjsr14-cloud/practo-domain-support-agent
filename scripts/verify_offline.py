"""Rebuild and demonstrate Part 1 with outbound Python sockets denied."""

import os
from pathlib import Path
import socket
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(name, None)
    os.environ["MOCK_LLM"] = "true"
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append("outbound network attempted")
        raise AssertionError("Offline verification forbids outbound sockets and DNS")

    with patch.object(socket.socket, "connect", blocked), \
            patch.object(socket.socket, "connect_ex", blocked), \
            patch.object(socket, "create_connection", blocked), \
            patch.object(socket, "getaddrinfo", blocked):
        # Import dependencies inside the guard, including telemetry initialization.
        from rag.index import RAGIndex
        from rag.evaluate import run_evaluation
        from scripts.calibrate_threshold import calibrate
        from scripts.run_part1_demo import run_demo
        from scripts.stress_threshold import run_stress
        from scripts.compare_retrieval import run_comparison

        print("API-key environment variables cleared; MOCK_LLM=true")
        print("Outbound Python sockets and DNS blocked before framework imports")
        # A fresh store proves that offline operation does not depend on an old index.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            index = RAGIndex(store_path=Path(directory) / "chroma")
            print(f"Fresh offline build: {index.build()}")
            calibration_path = Path(directory) / "calibration.json"
            calibrate(index, calibration_path)
            report = run_evaluation(index, persist=False)
            run_demo(index, strategy=report["recommended_strategy"],
                     calibration_path=calibration_path)
            run_stress(index, calibration_path=calibration_path)
            run_comparison(index, persist=False)
    if attempts:
        raise AssertionError(f"Unexpected outbound attempts: {len(attempts)}")
    print(f"Outbound Python network attempts: {len(attempts)}")
    print("Offline verification: PASS")


if __name__ == "__main__":
    main()
