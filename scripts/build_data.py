"""Build a standalone explorer from a directory of items."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from embed import IMAGES, TEXT, encode_collection, load_image, read_collection

ROOT = Path(__file__).resolve().parents[1]


def thumbnail(path, width):
    image = load_image(path)
    image.thumbnail((width, width * 2))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=72, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def float_base64(array):
    return base64.b64encode(array.astype("<f4").tobytes()).decode("ascii")


def project_space(vectors, vocab, dimensions, steps):
    vectors = vectors.astype(np.float64)
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    basis = components[:dimensions].copy()
    scores = centered @ basis.T
    for axis in range(dimensions):
        if scores[np.argmax(np.abs(scores[:, axis])), axis] < 0:
            basis[axis] *= -1
            scores[:, axis] *= -1
    variance = singular_values ** 2
    ratio = variance[:dimensions] / (variance.sum() or 1)

    layouts = {}
    if len(vectors) >= 3:
        from umap import UMAP

        previous = None
        for k in steps:
            if np.max(np.abs(scores[:, :k])) < 1e-10:
                layout = np.zeros((len(vectors), 2))
            else:
                layout = UMAP(n_neighbors=min(8, len(vectors) - 1), min_dist=0.3,
                              random_state=7, init="random", n_jobs=1).fit_transform(scores[:, :k])
                layout = (layout - layout.mean(axis=0)) / np.maximum(layout.std(axis=0), 1e-9)
            if previous is not None:
                # Align successive layouts so switching k can animate smoothly.
                left, _, right = np.linalg.svd(layout.T @ previous)
                layout = layout @ (left @ right)
            previous = layout
            layouts[k] = np.round(layout, 4).tolist()

    return dict(scores=np.round(scores, 5).tolist(), evr=np.round(ratio, 5).tolist(),
                cos=np.round(vectors @ vectors.T, 4).tolist(), umap=layouts,
                Vt=float_base64(basis), mean=float_base64(mean),
                vocab_a=np.round(vocab @ basis.T, 5).tolist(),
                vocab_c=np.round(vocab @ mean, 5).tolist())


def build(root, output):
    items = list(read_collection(root))
    if not items:
        raise ValueError(f"No supported files in {root}. Supported extensions: {', '.join(sorted(TEXT | IMAGES))}")
    vectors, terms, vocab = encode_collection(items)
    for name, rows in vectors.items():
        if rows.shape != (len(items), 768) or not np.isfinite(rows).all():
            raise ValueError(f"Invalid EmbeddingGemma 2 output for the {name} space.")
    if vocab.shape != (len(terms), 768) or not np.isfinite(vocab).all():
        raise ValueError("Invalid EmbeddingGemma 2 output for the search vocabulary.")

    dimensions = max(1, min(len(items) - 1, 768))
    steps = sorted({min(dimensions, k) for k in (2, 3, 5, 10, 20, 32, dimensions)})
    projects = []
    for name, title, body, images in items:
        projects.append(dict(slug=name, title=title, summary=body[:420],
                             kind="mixed" if body and images else "image" if images else "text",
                             thumb=thumbnail(images[0], 220) if images else None,
                             gallery=[thumbnail(path, 440) for path in images[1:5]]))
    data = dict(projects=projects, vocab=terms, steps=steps,
                spaces={name: project_space(rows, vocab, dimensions, steps) for name, rows in vectors.items()})
    data["id"] = hashlib.sha256(vectors["combined"].tobytes() + repr([item[0] for item in items]).encode()).hexdigest()[:16]
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False)
    page = (ROOT / "template.html").read_text(encoding="utf-8").replace("__DATA__", payload.replace("<", "\\u003c"))

    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        np.savez_compressed(stage / "embeddings.npz", slugs=np.array([item[0] for item in items]), **vectors)
        (stage / "data.json").write_text(payload, encoding="utf-8")
        (stage / "index.html").write_text(page, encoding="utf-8")
        for name in ("embeddings.npz", "data.json", "index.html"):
            (stage / name).replace(output / name)
    print(f"Built {output / 'index.html'} with {len(items)} items.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=ROOT / "input", help="collection directory (default: input/)")
    parser.add_argument("--output", type=Path, default=ROOT / "dist", help="output directory (default: dist/)")
    args = parser.parse_args()
    root, output = args.input.resolve(), args.output.resolve()
    if not root.is_dir():
        parser.error(f"Directory not found: {root}. Create it and add your files or project folders.")
    if output.is_relative_to(root):
        parser.error("Choose an output directory outside the input directory.")
    try:
        build(root, output)
    except ValueError as error:
        parser.exit(1, f"Build failed: {error}\n")


if __name__ == "__main__":
    main()
