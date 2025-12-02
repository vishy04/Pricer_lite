"""
Convenience module for loading and processing Amazon product data.
Provides a simple API for loading products by category.
"""

from parallel_loader import ItemLoader
from typing import List
from items import Item


def process_products_for_category(category_name: str, workers: int = 8) -> List[Item]:
    """
    Load and process products for a specific category.
    
    Args:
        category_name: Category name (e.g., "books", "beauty", "electronics")
                      Must exist as raw_meta_{category_name} in the dataset
        workers: Number of parallel workers (default: 8)
    
    Returns:
        List of processed Item objects with item.include == True
    
    Examples:
        >>> items = process_products_for_category("books")
        >>> print(len(items), "clean items")
        >>> sample = items[0]
        >>> print("Tokens:", sample.token_count)
        >>> print("Prompt snippet:", sample.prompt[:300])
    """
    loader = ItemLoader(category_name)
    return loader.load_and_process_data(workers=workers)