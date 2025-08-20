from datetime import datetime
from typing import List, Optional

from datasets import load_dataset
from tqdm import tqdm

from items import Item


class CategoryLoader:
    """
    Loading and filterering metadata for a specific category and build Item objects.
    """

    # Default price range filter
    DEFAULT_MIN_PRICE: float = 0.50
    DEFAULT_MAX_PRICE: float = 999.49

    def __init__(self, category_name: str, *, min_price: Optional[float] = None, max_price: Optional[float] = None):
        self.category_name = category_name
        self.min_price = self.DEFAULT_MIN_PRICE if min_price is None else float(min_price)
        self.max_price = self.DEFAULT_MAX_PRICE if max_price is None else float(max_price)
        self._dataset = None

    def _load_dataset(self):
        """Load the HF dataset split for this category into memory (cached by HF)."""
        return load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{self.category_name}",
            split="full",
            trust_remote_code=True,
        )

    @staticmethod
    def _parse_price(value) -> Optional[float]:
        """Convert a price-like value to float; return None if invalid."""
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _price_in_range(self, price: float) -> bool:
        return self.min_price <= price <= self.max_price

    def _build_items(self) -> List[Item]:
        """Iterate rows, validate price, construct Item objects, and filter by Item.include."""
        clean_items: List[Item] = []
        for data_point in tqdm(self._dataset, desc=f"   - Filtering {self.category_name}"):
            price = self._parse_price(data_point.get("price"))
            if price is None or not self._price_in_range(price):
                continue

            item = Item(data_point, price)
            if item.include:
                item.category = self.category_name
                clean_items.append(item)

        return clean_items

    def process(self) -> List[Item]:
        """
        End-to-end: load dataset → filter/transform → return List[Item].
        Prints a short runtime summary.
        """
        start_time = datetime.now()
        try:
            self._dataset = self._load_dataset()
        except Exception:
            # If the dataset cannot be loaded, return an empty list silently (caller can handle)
            return []

        clean_items = self._build_items()

        duration_minutes = (datetime.now() - start_time).total_seconds() / 60
        print(f"{len(clean_items):,} items for '{self.category_name}'.")
        print(f"time: {duration_minutes:.1f} minutes.")
        return clean_items


# Backward-compatible helper 
def process_products_for_category(category_name: str) -> List[Item]:
    loader = CategoryLoader(category_name)
    return loader.process()



