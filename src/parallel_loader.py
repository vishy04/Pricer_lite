import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from datetime import datetime
import math
from datasets import load_dataset  # hugging face dataset
from tqdm import tqdm
from items import Item

MIN_PRICE = 0.5
MAX_PRICE = 999.49
CHUNK_SIZE = 1000
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

class ItemLoader:

    def __init__(self,name):
        self.name = name 
        self.dataset = None #will be assigned once the dataset is loaded
    
    def create_item_from_datapoint(self, datapoint):
        '''
        Converts raw datapoints into Item objects , within a price range
        returns None if items should be excluded
        '''
        try:
            price_str = datapoint.get('price')
            if price_str:
                price = float(price_str)
                if MIN_PRICE <= price <= MAX_PRICE:
                    item = Item(datapoint, price)
                    return item if item.include else None
        except (ValueError, TypeError):
            return None
    
    def process_chunk(self,chunks):
        '''
        creates a batch list from processed datapoint
        '''
        batch=[]
        for datapoint in chunks:
            result = self.create_item_from_datapoint(datapoint)
            if result :
                batch.append(result)
        return batch
    
    def chunk_iterator(self):
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield list(self.dataset.select(range(i, min(i + CHUNK_SIZE, size))))


def _process_chunk_worker(chunk, min_price: float, max_price: float, category_name: str):
    batch = []
    for datapoint in chunk:
        try:
            price_str = datapoint.get('price')
            if not price_str:
                continue
            price = float(price_str)
        except (ValueError, TypeError):
            continue
        if min_price <= price <= max_price:
            item = Item(datapoint, price)
            if item.include:
                item.category = category_name
                batch.append(item)
    return batch
    
    def load_in_parallel(self, workers):
        results = []
        chunk_count = math.ceil(len(self.dataset) / CHUNK_SIZE)
        worker = partial(_process_chunk_worker, min_price=MIN_PRICE, max_price=MAX_PRICE, category_name=self.name)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(pool.map(worker, self.chunk_iterator(), chunksize=1), total=chunk_count):
                results.extend(batch)
        return results
            
    def load_and_process_data(self , workers = 8): #default is 8 for mac m1
        start = datetime.now()
        print(f"Loading dataset {self.name}", flush=True)
        self.dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                                     f"raw_meta_{self.name}", 
                                     split="full", 
                                     trust_remote_code=True)
        results = self.load_in_parallel(workers) 
        finish = datetime.now() 
        minutes = (finish - start).total_seconds() / 60
        print(f"Completed {self.name} with {len(results):,} datapoints in {minutes:.1f} mins", flush=True)
        return results 
             

    

