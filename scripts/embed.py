"""Read local collections and encode their text and images."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODEL_ID = "google/embeddinggemma-2"
TEXT = {".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json"}
IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
MEDIA_TOKENS = ("<|image|>", "<|video|>", "<|audio|>")
# Leave room for the title and two images even in byte-heavy text.
CHUNK_SIZE = 1600


def read_collection(root):
    for item in sorted(root.iterdir()):
        if item.name.startswith(".") or item.is_symlink():
            continue
        if item.is_dir():
            files = []
            for folder, dirs, names in item.walk():
                dirs[:] = [name for name in dirs if not name.startswith(".")]
                files.extend(folder / name for name in names)
        else:
            files = [item]

        texts, images = [], []
        for path in sorted(files):
            if path.name.startswith(".") or path.is_symlink() or not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in TEXT:
                try:
                    value = path.read_text(encoding="utf-8-sig").strip()
                except UnicodeDecodeError as error:
                    raise ValueError(f"Expected UTF-8 text in {path}") from error
                if value:
                    texts.append(value)
            elif suffix in IMAGES:
                images.append(path)

        if texts or images:
            yield item.name, item.name if item.is_dir() else item.stem, "\n\n".join(texts), images


def load_image(path):
    try:
        with Image.open(path) as source:
            source.draft("RGB", (768, 768))
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((768, 768))
            return image
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError(f"Could not read image {path}") from error


def unit_mean(vectors):
    mean = np.mean(vectors, axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm else mean


def encode_collection(items):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    config = {"audio_config": None}
    if not any(item[3] for item in items):
        config["vision_config"] = None
    model = SentenceTransformer(MODEL_ID, model_kwargs={"torch_dtype": torch.float32}, config_kwargs=config)
    vectors = {"text": [], "image": [], "combined": []}
    for name, title, body, paths in items:
        print(f"Embedding {name}", flush=True)
        images = [load_image(path) for path in paths]
        kind_text = body
        for token in MEDIA_TOKENS:
            kind_text = kind_text.replace(token, " ")
            title = title.replace(token, " ")
        chunks = [kind_text[i:i + CHUNK_SIZE] for i in range(0, len(kind_text), CHUNK_SIZE)] or [""]
        documents = [f"title: {title} | text: {chunk}" for chunk in chunks]
        text_vectors = model.encode(documents, batch_size=8, normalize_embeddings=True)
        vectors["text"].append(unit_mean(text_vectors))
        if images:
            image_vectors = model.encode(images, batch_size=8, normalize_embeddings=True)
            vectors["image"].append(unit_mean(image_vectors))
            first = images[:2]
            mixed = model.encode([{
                "text": documents[0] + " " + " ".join(["<|image|>"] * len(first)),
                "image": first,
            }], normalize_embeddings=True)
            vectors["combined"].append(unit_mean(np.concatenate([mixed, text_vectors[1:]])))
        else:
            vectors["combined"].append(vectors["text"][-1])

    if len(vectors["image"]) != len(items):
        del vectors["image"]
    terms = list(dict.fromkeys(Path(__file__).with_name("vocab.txt").read_text().splitlines()))
    vocab = model.encode(terms, prompt_name="SearchQuery", batch_size=32, normalize_embeddings=True)
    return {name: np.asarray(rows) for name, rows in vectors.items()}, terms, vocab
