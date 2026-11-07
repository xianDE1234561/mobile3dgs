"""Real local HTTP: failed transfers must not poison the reusable dataset cache."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
from pathlib import Path
import tarfile
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

from mobile3dgs.data import SEQUENCE, fetch


class FetchCheck(unittest.TestCase):
    def test_failed_download_can_be_retried(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
                content = b"0 example\n"
                member = tarfile.TarInfo(f"{SEQUENCE}/{name}")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        complete = buffer.getvalue()
        bodies = iter([(complete[:len(complete)//2], len(complete)),
                       (b"not a gzip", 10), (complete, len(complete))])

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body, length = next(bodies)
                self.send_response(200)
                self.send_header("Content-Length", str(length))
                self.end_headers()
                self.wfile.write(body)

        with tempfile.TemporaryDirectory() as directory, ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                root = Path(directory)
                cached = root / f"{SEQUENCE}.tgz"
                with patch("mobile3dgs.data.URL", f"http://127.0.0.1:{server.server_port}/desk"):
                    for _ in range(2):
                        with self.assertRaises(ValueError):
                            fetch(root)
                        self.assertFalse(cached.exists())
                        self.assertFalse((root / SEQUENCE).exists())
                    # A previously poisoned cache is also recoverable without manual deletion.
                    cached.write_bytes(b"old invalid archive")
                    with self.assertRaises(ValueError):
                        fetch(root)
                    self.assertFalse(cached.exists())
                    result = fetch(root)
                self.assertEqual(cached.read_bytes(), complete)
                self.assertEqual(result["archive_bytes"], len(complete))
                self.assertEqual((root / SEQUENCE / "rgb.txt").read_bytes(), b"0 example\n")
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    unittest.main()
