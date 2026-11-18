import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_data import build
from embed import read_collection


class CollectionTests(unittest.TestCase):
    def test_groups_nested_files_and_skips_hidden_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "A project" / "nested").mkdir(parents=True)
            (root / "A project" / "nested" / "notes.md").write_text("Project description")
            (root / "A project" / ".hidden.txt").write_text("Private")
            (root / "A project" / "ignored.pdf").touch()
            image = root / "A project" / "cover.png"
            Image.new("RGB", (10, 10)).save(image)
            (root / "Loose.txt").write_text("Standalone notes")

            self.assertEqual(list(read_collection(root)), [
                ("A project", "A project", "Project description", [image]),
                ("Loose.txt", "Loose", "Standalone notes", []),
            ])

    def test_single_item_exports_finite_data_and_escapes_html(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            output = Path(directory) / "output"
            root.mkdir()
            (root / "Notes.txt").write_text("</script><script>alert(1)</script>")
            vector = np.eye(1, 768, dtype=np.float32)
            result = {"text": vector, "combined": vector}, ["nature"], vector
            with patch("build_data.encode_collection", return_value=result):
                build(root, output)

            data = json.loads((output / "data.json").read_text())
            self.assertEqual(data["spaces"]["combined"]["scores"], [[0.0]])
            self.assertEqual(data["spaces"]["combined"]["umap"], {})
            html = (output / "index.html").read_text()
            payload = html.split('id="data" type="application/json">')[1].split("</script>")[0]
            self.assertNotIn("<script>alert(1)", payload)
            self.assertEqual(json.loads(payload), data)
            with np.load(output / "embeddings.npz") as archive:
                self.assertEqual(archive["slugs"].tolist(), ["Notes.txt"])


if __name__ == "__main__":
    unittest.main()
