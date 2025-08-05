import os 
from pathlib import Path
#directory setup 
directories = [
        "Pricer",
        "Pricer/data",
        "Pricer/data/raw",
        "Pricer/data/processed",
        "Pricer/src",
        "Pricer/notebooks",
        "Pricer/models",
        "Pricer/models/baseline",
        "Pricer/models/frontier",
        "Pricer/models/fine_tuned",
        "Pricer/results",
        "Pricer/results/charts",
        "Pricer/results/metrics",
        "Pricer/results/reports",
        "Pricer/docs"
    ]

for directory in directories :
    Path(directory).mkdir(parents=True , exist_ok=True)
    print(f"Created :{directory}")