# Pricer Project

## How to Use the Item Class

The `Item` class in `src/items.py` processes product data and creates training prompts for price prediction.

### Basic Usage

```python
import sys
sys.path.append('../src')
from items import Item

# Sample data structure
sample_data = {
    'title': 'Premium Wireless Bluetooth Headphones',
    'category': 'Electronics',
    'description': ['High-quality wireless headphones with noise cancellation'],
    'features': ['Bluetooth 5.0', 'Active noise cancellation'],
    'details': 'Package includes: Headphones, USB-C cable, Carry case'
}

# Create Item object
item = Item(sample_data, price=129.99)

# Check if item meets criteria for training
if item.include:
    print(f"Training prompt: {item.prompt}")
    print(f"Token count: {item.token_count}")
    print(f"Test prompt: {item.get_test_prompt()}")
else:
    print("Item excluded - insufficient content")
```

### Key Features

1. **Automatic Processing**: Combines title, description, features, and details
2. **Noise Removal**: Removes irrelevant text like battery info, manufacturer details
3. **Length Validation**: Ensures sufficient content (800+ characters)
4. **Token Management**: Truncates to fit within token limits (150-180 tokens)
5. **Prompt Generation**: Creates standardized training and test prompts

### Data Requirements

The Item class expects a dictionary with these keys:

- `title` (string): Product title
- `description` (list): Product description paragraphs
- `features` (list): Product features
- `details` (string): Additional product details
- `category` (string, optional): Product category

### Processing Criteria

- **Minimum Characters**: 800 (ensures sufficient content)
- **Maximum Characters**: 1500 (prevents token overflow)
- **Minimum Tokens**: 150 (ensures meaningful content for training)
- **Maximum Tokens**: 180 (keeps training efficient)

### Example Output

```python
# Training prompt
"How much does this cost to the nearest dollar?

Premium Wireless Bluetooth Headphones
High-quality wireless headphones with noise cancellation
Bluetooth 5.0, Active noise cancellation
Package includes: Headphones, USB-C cable, Carry case

Price is $130.00"

# Test prompt (for inference)
"How much does this cost to the nearest dollar?

Premium Wireless Bluetooth Headphones
High-quality wireless headphones with noise cancellation
Bluetooth 5.0, Active noise cancellation
Package includes: Headphones, USB-C cable, Carry case

Price is $"
```

### Batch Processing

For large datasets, use the batch processing function:

```python
def process_items_batch(dataset, batch_size=1000):
    """Process items in batches to avoid memory issues"""
    all_items = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        for datapoint in batch:
            try:
                price = float(datapoint["price"])
                if price > 0:
                    item = Item(datapoint, price)
                    if item.include:
                        all_items.append(item)
            except ValueError:
                pass
    return all_items
```
