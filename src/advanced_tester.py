import math
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# COLOR MAP
GREEN = "\033[92m"
ORANGE = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": ORANGE, "green": GREEN}

class Tester_v2:
    def __init__(self, predictor, data, title=None, size=250):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size
        self.guesses = []
        self.truths = []
        self.errors = []
        self.lche = []
        self.sles = []
        self.colors = []
        # New counters for progress tracking
        self.green_count = 0
        self.orange_count = 0
        self.red_count = 0

    def run_datapoint(self, i):
        datapoint = self.data.iloc[i]

        guess = float(self.predictor(datapoint))  # predicted output
        truth = float(datapoint['price'])  # always be positive

        error = abs(truth - guess)
        log_error = math.log(truth + 1) - math.log(guess + 1)
        sle = log_error ** 2
        log_cosh_error = self.safe_log_cosh(error)

        color = self.color_for(error, truth)  # for better outputs
        
        # Update counters based on color
        if color == 'green':
            self.green_count += 1
        elif color == 'orange':
            self.orange_count += 1
        else:  # red
            self.red_count += 1

        # Store all results
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.colors.append(color)
        self.lche.append(log_cosh_error)
        self.sles.append(sle)

        return color

    def safe_log_cosh(self, x):
        """avoids overflow"""
        x = max(min(x, 500), -500)  # Cap between -500 and 500
        return math.log(math.cosh(x))

    def color_for(self, error, truth):
        if error < 40 or error/truth < 0.2:
            return 'green'
        elif error < 80 or error/truth < 0.4:
            return 'orange'
        else:
            return 'red'

    def chart(self, title):
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Get max value for axes
        max_val = max(max(self.truths), max(self.guesses))
        
        # Add error bands (±20% and ±40%)
        x = np.linspace(0, max_val, 100)
        plt.fill_between(x, x*0.8, x*1.2, color='green', alpha=0.1, label='±20% Range')
        plt.fill_between(x, x*0.6, x*1.4, color='orange', alpha=0.1, label='±40% Range')
        
        # Perfect prediction line
        plt.plot([0, max_val], [0, max_val], color='skyblue', lw=2, 
                alpha=0.6, label='Perfect Prediction')
        
        # Scatter points with increased size and transparency
        plt.scatter(self.truths, self.guesses, s=50, c=self.colors, 
                alpha=0.6, label='Predictions')
        
        # Add statistics text box
        average_error = sum(self.errors) / self.size
        rmsle = math.sqrt(sum(self.sles)/self.size)
        green_pct = (self.colors.count('green')/self.size*100)
        orange_pct = (self.colors.count('orange')/self.size*100)
        red_pct = (self.colors.count('red')/self.size*100)
        
        stats_text = (
            f'Accuracy Distribution:\n'
            f'Green: {green_pct:.1f}%\n'
            f'Orange: {orange_pct:.1f}%\n'
            f'Red: {red_pct:.1f}%\n'
            f'Average Error: ${average_error:,.2f}\n'
            f'RMSLE: {rmsle:.2f}'
        )
        
        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Customize plot
        plt.xlabel('True Values ($)', fontsize=12)
        plt.ylabel('Predicted Values ($)', fontsize=12)
        plt.title(title, fontsize=14, pad=20)
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        
        # Display
        plt.tight_layout()
        plt.show()

    def report(self):
        average_error = sum(self.errors) / self.size
        rmsle = math.sqrt(sum(self.sles)/self.size)
        hit_rate = (self.green_count / self.size) * 100
        
        # Print summary statistics with color
        print("\nTest Results Summary:")
        print(f"Total Predictions: {self.size}")
        print(f"{GREEN}Correct (Green): {self.green_count} ({self.green_count/self.size*100:.1f}%){RESET}")
        print(f"{ORANGE}Close (Orange): {self.orange_count} ({self.orange_count/self.size*100:.1f}%){RESET}")
        print(f"{RED}Wrong (Red): {self.red_count} ({self.red_count/self.size*100:.1f}%){RESET}")
        print(f"Average Error: ${average_error:,.2f}")
        print(f"RMSLE: {rmsle:,.2f}")
        
        title = f"{self.title} Error=${average_error:,.2f}  RMSLE={rmsle:,.2f}  HIT={hit_rate:.1f}%"
        self.chart(title)

    def run(self):
        self.error = 0
        
        # Progress bar for overall testing
        with tqdm(total=self.size, desc="Testing Progress", 
                 bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            
            # Progress bar for correct predictions
            with tqdm(total=self.size, desc="Correct Predictions", 
                     bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}") as correct_pbar:
                
                for i in range(self.size):
                    color = self.run_datapoint(i)
                    pbar.update(1)
                    
                    # Update correct predictions bar based on color
                    if color == 'green':
                        correct_pbar.update(1)
        
        self.report()

    @classmethod
    def test(cls, function, data):
        cls(function, data).run()