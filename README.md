# emex

Explore a folder of text and images as an interactive embedding map. emex runs
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) locally, then
builds a standalone HTML explorer with 3D and 2D PCA, UMAP, parallel
coordinates, axis labels, and nearest neighbors.

## Quick start

Use Python 3.12 or later. From the repository directory, create a virtual
environment and install the dependencies.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate it with `.venv\Scripts\Activate.ps1`.

Drop project folders or individual files into `input/`. Each immediate
subfolder becomes one point using its supported files, including files in nested
folders. Each supported file directly under `input/` becomes its own point.

```text
input/
  Forest trail/
    notes.md
    canopy.jpg
  City garden/
    notes.md
    plans/
      planting.png
  Field notes.txt
```

```sh
python scripts/build_data.py
```

Open `dist/index.html` in a browser. The first build downloads the model.
Run the same command after adding, editing, or removing inputs. To try the
included example or use a directory elsewhere:

```sh
python scripts/build_data.py examples/field-notes --output dist/example
python scripts/build_data.py "/path/to/my collection" --output dist/my-collection
```

### Supported files

- UTF-8 text · TXT, MD, MARKDOWN, RST, CSV, TSV, and JSON
- Images · JPEG, PNG, WebP, GIF, BMP, and TIFF. Animated files use their first frame.

Hidden files, symlinks, and unsupported formats are skipped. For video, you
can place extracted still frames in the project folder. For PDFs or office
documents, export the text to a supported format.

### Explore

- **Overall** uses the title and text with up to two images in a shared model input.
  Additional text is embedded in chunks. **Text** uses the title and text.
  **Image** averages all image embeddings and is available when every item has an image.
- **3D and 2D** show PCA projections. Scroll to cycle axes, select an axis chip
  to assign it to X, Y, or Z, and click an item to see its neighbors.
- **UMAP** shows local clusters and is available with at least three items.
  **Parallel** shows per-axis scores, and **Axes** shows the items and vocabulary
  terms at each end of an axis.
- **Search axes** ranks items by a term or a contrast such as `speculative vs practical`.
  Built-in terms work offline. Free-text queries download a separate query model
  and run in a browser with WebGPU. **Pin** keeps a search direction and
  recomputes PCA for the remaining variation.

The build also writes `dist/data.json` for the viewer and `dist/embeddings.npz`
for analysis. Input files and generated output stay out of Git by default.
A shared HTML export includes item names, text excerpts, thumbnails, galleries,
and derived data. Review the export before sharing it.

## Customize

See [the AI agent guide](AGENTS.md) for the pipeline, extension points, and a
starter prompt.
