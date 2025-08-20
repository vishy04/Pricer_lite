from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from items import Item

# We'll only process items within this price range.
MIN_PRICE = 0.50
MAX_PRICE = 999.49

def process_products_for_category(category_name):
    """
    Loads raw product data for a category, filters out unwanted items,
    and returns a clean list of Item objects.
    """
    start_time = datetime.now()

    # Load the dataset from the hub.
    try:
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category_name}",
            split="full",
            trust_remote_code=True
        )
    except Exception as e:
        return []

    clean_items = []
    # Loop through and validate each item.
    for data_point in tqdm(dataset, desc="   - Filtering"):
        try:
            price_str = data_point.get('price')
            if not price_str:
                continue

            price = float(price_str)

            if MIN_PRICE <= price <= MAX_PRICE:
                item = Item(data_point, price)
                if item.include:
                    item.category = category_name
                    clean_items.append(item)

        except (ValueError, TypeError):
            # Price not a valid number-skip it.
            continue

    end_time = datetime.now()
    duration_minutes = (end_time - start_time).total_seconds() / 60

    print(f"{len(clean_items):,} items for '{category_name}'.")
    print(f"time: {duration_minutes:.1f} minutes.")

    return clean_items



