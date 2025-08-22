## Pricer Lite

A streamlined data preparation pipeline to load Amazon product metadata, filter/clean it, and build model-ready prompts for price-prediction tasks.

### What it does

- Loads raw product metadata from the Hugging Face dataset `McAuley-Lab/Amazon-Reviews-2023`
- Filters items by price range and content quality
- Cleans text (title, description, features, details)
- Builds training/test prompts capped to token limits using a Llama tokenizer
- Displays progress bars and basic run timing

---

## Project structure

```
Pricer_lite/
├─ src/
│  ├─ __init__.py
│  ├─ items.py           # Item class: cleaning, tokenization, prompt creation
│  ├─ parallel_loader.py # ItemLoader class for parallel processing
│  └─ loaders.py         # Convenience functions: process_products_for_category()
├─ notebooks/
│  ├─ 1.data_investigation.ipynb # Data exploration and analysis
│  └─ 2.data_loading.ipynb       # Parallel loading examples
├─ results/
│  └─ charts/            # Generated visualizations and plots
├─ requirements.txt      # Python dependencies
├─ environment.yml       # Conda environment specification
├─ dir_setup.py         # Script to create full project structure
├─ test_api.ipynb       # API testing notebook
└─ README.md
```

---

## Installation

Use either Conda or venv. Ensure you activate your working environment before installing and running anything.

**Conda (recommended):**

```bash
cd Pricer_lite
conda env create -f environment.yml
conda activate pricer
```

**venv:**

```bash
cd Pricer_lite
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Note:** First run will download the tokenizer and dataset cache to your user cache directory (~/.cache/huggingface/).

---

## Quick start

### Python script / REPL

```python
import os, sys
sys.path.append(os.path.abspath("src"))

from loaders import process_products_for_category

# Examples of categories: "books", "beauty", "electronics" (must exist as raw_meta_{category})
items = process_products_for_category("books")
print(len(items), "clean items")

sample = items[0]
print(sample)
print("Tokens:", sample.token_count)
print("Prompt snippet:\n", sample.prompt[:300])
```

### Jupyter notebook (from `Pricer_lite/notebooks`)

```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))

from loaders import process_products_for_category
items = process_products_for_category("books")
len(items)
```

**Alternative - Direct ItemLoader usage:**

```python
from parallel_loader import ItemLoader
loader = ItemLoader("books")
items = loader.load_and_process_data(workers=8)
```

---

## Core modules

### `src/items.py` — Item

Creates a cleaned, bounded prompt with the product’s text and price, using a Llama tokenizer for token limits.

Key attributes:

- `title: str`
- `price: float`
- `category: str`
- `prompt: str | None`
- `token_count: int`
- `include: bool` (True only if all quality checks pass)

Important methods:

- `get_test_prompt()` → returns the prompt with the price masked (for evaluation/inference)

Main limits (edit in `src/items.py`):

- `MODEL` (tokenizer)
- `MIN_TOKENS`, `MAX_TOKENS`
- `MIN_CHARS`, `MAX_CHARS`
- `NOISE` (strings removed from details)

Minimal example:

```python
from items import Item

sample_data = {
    'title': 'Premium Wireless Bluetooth Headphones',
    'category': 'Electronics',
    'description': ['High-quality wireless headphones with noise cancellation'],
    'features': ['Bluetooth 5.0', 'Active noise cancellation'],
    'details': 'Package includes: Headphones, USB-C cable, Carry case'
}

item = Item(sample_data, price=129.99)
if item.include:
    print(item.prompt)
    print(item.get_test_prompt())
```

### `src/loaders.py` — Convenience functions

**Main entrypoint:** `process_products_for_category(category_name: str, workers: int = 8) -> List[Item]`

Simple wrapper around the parallel loader for easy usage.

### `src/parallel_loader.py` — ItemLoader class

**Core class:** `ItemLoader(category_name: str)`

**Main method:** `load_and_process_data(workers: int = 8) -> List[Item]`

What it does:

- Loads split `raw_meta_{category_name}` from `McAuley-Lab/Amazon-Reviews-2023`
- Filters by a price range
- Builds `Item` objects in parallel and keeps only those where `item.include == True`

**Config constants:**

- `MIN_PRICE = 0.50`
- `MAX_PRICE = 999.49`
- `CHUNK_SIZE = 1000`

---

## Additional Features

### Notebooks

- **`1.data_investigation.ipynb`** — Data exploration, analysis, and visualization of the Amazon dataset
- **`2.data_loading.ipynb`** — Examples of parallel data loading and processing with multiple categories

### Visualization Results

The `results/charts/` directory contains generated plots:
- Price distributions (`BPrice.png`, `Final Price.png`)
- Token analysis (`tokens.png`, `Price vs Token.png`)
- Content length analysis (`Lenghts Plot.png`)

### Setup Utilities

- **`dir_setup.py`** — Script to create the full project structure for larger implementations
- **`test_api.ipynb`** — Notebook for testing API functionality

---

## Performance, caching, and tips

- Progress bars via `tqdm`
- Tokenizer and datasets are cached under your Hugging Face cache (e.g., `~/.cache/huggingface/...`)
- Start with a smaller category (e.g., `beauty`) to validate setup quickly
- If you hit memory pressure, process a subset first or run in a fresh environment

---

## Troubleshooting

**ImportError in notebooks**

```python
# Add src to Python path at the top of your notebook
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))
```

**Tokenizer / dataset download issues**

- Ensure internet access and that your environment can reach `huggingface.co`
- Re-run after transient failures; caches make subsequent runs faster

**Invalid category**

- The category must exist as a dataset config: `raw_meta_{category_name}`

**Environment conflicts**

- Create a clean environment using the provided `environment.yml` or venv

---

## Acknowledgements

- Dataset: `McAuley-Lab/Amazon-Reviews-2023` (Hugging Face)
- Tokenizer and tooling: `transformers`, `datasets`, `tqdm`
