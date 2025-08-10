import re
from transformers import AutoTokenizer
from typing import Optional

class Item:
    """
    Cleans product data and creates training prompts.
    """
    
    # Model
    MODEL = "meta-llama/Meta-Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    
    # Token limits for model
    MIN_TOKENS = 150  
    MAX_TOKENS = 180  
    
    # Character limits (rought 7*Token -200 ( from Length Plot ) for MIN)
    MIN_CHARS = 800   
    MAX_CHARS = 1500 #test
    
    # Prompt template
    QUESTION = "How much does this cost to the nearest dollar?"
    PREFIX = "Price is $"
    
    #REMOVAL -> SPECIALLY IN THE DETAILS
    NOISE = [
        '"Batteries Included?": "No"', '"Batteries Included?": "Yes"',
        '"Batteries Required?": "No"', '"Batteries Required?": "Yes"',
        "By Manufacturer", "Item", "Date First", "Package", ":",
        "Number of", "Best Sellers", "Number", "Product "
    ]
    
    def __init__(self, data: dict, price: float):

        self.title = data.get('title', '')
        self.price = price
        self.category = data.get('category', '')
        self.prompt = None
        self.token_count = 0
        self.include = False
        
        #Immediate processing
        self._process_data(data)
    
    def _process_data(self, data: dict):
        """
        Main Processing: Combine content, validate length, tokenize, create prompt
        """
        # 1. Combine , clean text content, limit the length 
        content = self._combine_content(data)
        if len(content) < self.MIN_CHARS:
            return  # Skip if insufficient context 
        content = content[:self.MAX_CHARS]
        
        # 2. Tokenize and validate token limits
        full_text = f"{self._clean_text(self.title)}\n{self._clean_text(content)}"
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        if len(tokens) < self.MIN_TOKENS:
            return  # Skip items with insufficient tokens
        tokens = tokens[:self.MAX_TOKENS]
        clean_text = self.tokenizer.decode(tokens)
        
        # 3. Create training prompt and mark for inclusion
        self._create_prompt(clean_text)
        self.include = True
    
    def _combine_content(self, data: dict) -> str:
        """
        Getting Richer Context.
        """
        parts = []
        
        # Add description if available
        if data.get('description'):
            parts.extend(data['description'])
        
        # Add features if available
        if data.get('features'):
            parts.extend(data['features'])
        
        # Add cleaned details if available
        if data.get('details'):
            parts.append(self._remove_noise(data['details']))
        
        return '\n'.join(parts)
    
    def _remove_noise(self, text: str) -> str:
        """
        Remove noise that don't help with pricing.
        """
        for pattern in self.NOISE:
            text = text.replace(pattern, "")
        return text
    
    def _clean_text(self, text: str) -> str:
        """
            removing special characters and filtering words.
        """
        # special characters and normalize whitespace[using regex is faster and better]
        text = re.sub(r'[:\[\]"{}【】\s]+', ' ', text).strip()
        
        # punctuation issues
        text = text.replace(" ,", ",").replace(",,,", ",").replace(",,", ",")
        
        # Filter words: keep short words or words without numbers
        words = text.split(' ')
        filtered_words = [
            word for word in words 
            if len(word) < 7 or not any(char.isdigit() for char in word)
        ]
        
        return " ".join(filtered_words)
    
    def _create_prompt(self, text: str):
        """
        Create training prompt with question, content, and price.
        """
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n{self.PREFIX}{int(self.price)}.00"
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))
    
    def get_test_prompt(self) -> str:
        """
        Get prompt without price for testing i.e training prompt just without the price.
        """
        if self.prompt:
            return self.prompt.split(self.PREFIX)[0] + self.PREFIX
        return ""
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"<{self.title} = ${self.price}>" 