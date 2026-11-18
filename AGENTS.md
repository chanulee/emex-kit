# Reconfiguring emex with an AI agent

Start by describing what should count as an item, which formats the collection
uses, and what you want to compare. By default, each immediate subfolder or
standalone supported file under `input/` becomes one item.

> Adapt emex for my collection of [describe your data]. Treat each [folder or
> file] as one point, read [formats], and display [fields] when I select an
> item. Keep the explorer self-contained. Build and check it with the sample
> collection before using my own files.

## Where to make changes

| File | Responsibility |
| --- | --- |
| `scripts/embed.py` | Discover inputs, prepare text and images, and run EmbeddingGemma 2. Change item grouping and add format readers here. |
| `scripts/build_data.py` | Build PCA and UMAP, generate previews, and write `data.json`, `embeddings.npz`, and `index.html`. Change the item metadata and export schema here. |
| `template.html` | Render the explorer, including charts, item details, nearest neighbors, and search. Edit this source to change the interface. |
| `scripts/vocab.txt` | Supply built-in terms for offline search axes and automatic axis labels. Use one term per line. |

Keep item order aligned across the embedding matrices, PCA scores, and viewer
metadata. If you change the model, rebuild the item embeddings and
vocabulary together and update the browser query model and projection
dimension to match.

## Check a change

1. Install the requirements listed in the README.
2. Run `python scripts/build_data.py examples/field-notes --output dist/example`.
3. Open `dist/example/index.html`. Switch views, click an item, and search for
   `nature vs city`.
4. Repeat with a small representative subset of your own collection.
