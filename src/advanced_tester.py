import math
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

# COLOR MAP
GREEN = "\033[92m"
ORANGE = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": ORANGE, "green": GREEN}

class Tester:
    def __init__(self, predictor, data, title=None, size=250):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size
        
        # Use numpy arrays for numerical data
        self.guesses = np.zeros(size)
        self.truths = np.zeros(size)
        self.errors = np.zeros(size)
        self.lche = np.zeros(size)
        self.sles = np.zeros(size)
        self.colors = []  # Keep as list for strings
        
        # Counters
        self.green_count = 0
        self.orange_count = 0
        self.red_count = 0
        
        # Cache for computed metrics
        self._average_error = None
        self._rmsle = None

    def run_datapoint(self, i):
        try:
            datapoint = self.data.iloc[i]
            guess = float(self.predictor(datapoint))
            truth = float(datapoint['price'])

            error = abs(truth - guess)
            log_error = math.log(truth + 1) - math.log(guess + 1)
            sle = log_error ** 2
            log_cosh_error = self.safe_log_cosh(error)

            color = self.color_for(error, truth)
            
            # Update counters
            if color == 'green':
                self.green_count += 1
            elif color == 'orange':
                self.orange_count += 1
            else:  # red
                self.red_count += 1

            # Store results in arrays
            self.guesses[i] = guess
            self.truths[i] = truth
            self.errors[i] = error
            self.lche[i] = log_cosh_error
            self.sles[i] = sle
            self.colors.append(color)

            return color
            
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Error processing datapoint {i}: {e}")
            self.colors.append('red')
            self.red_count += 1
            return 'red'

    def safe_log_cosh(self, x):
        """Avoids overflow in log cosh calculation"""
        x = max(min(x, 500), -500)  # Cap between -500 and 500
        return math.log(math.cosh(x))

    def color_for(self, error, truth):
        if error < 40 or error/truth < 0.2:
            return 'green'
        elif error < 80 or error/truth < 0.4:
            return 'orange'
        else:
            return 'red'

    @property
    def average_error(self):
        if self._average_error is None:
            self._average_error = np.mean(self.errors)
        return self._average_error

    @property
    def rmsle(self):
        if self._rmsle is None:
            self._rmsle = math.sqrt(np.mean(self.sles))
        return self._rmsle

    def chart(self, title):
        plt.figure(figsize=(12, 8))
        
        max_val = max(np.max(self.truths), np.max(self.guesses))
        
        # Add error bands
        x = np.linspace(0, max_val, 100)
        plt.fill_between(x, x*0.8, x*1.2, color='green', alpha=0.1, label='±20% Range')
        plt.fill_between(x, x*0.6, x*1.4, color='orange', alpha=0.1, label='±40% Range')
        
        # Perfect prediction line
        plt.plot([0, max_val], [0, max_val], color='skyblue', lw=2, 
                alpha=0.6, label='Perfect Prediction')
        
        # Scatter plot
        plt.scatter(self.truths, self.guesses, s=50, c=self.colors, 
                   alpha=0.6, label='Predictions')
        
        # Statistics text box
        green_pct = (self.green_count/self.size*100)
        orange_pct = (self.orange_count/self.size*100)
        red_pct = (self.red_count/self.size*100)
        
        stats_text = (
            f'Accuracy Distribution:\n'
            f'Green: {green_pct:.1f}%\n'
            f'Orange: {orange_pct:.1f}%\n'
            f'Red: {red_pct:.1f}%\n'
            f'Average Error: ${self.average_error:,.2f}\n'
            f'RMSLE: {self.rmsle:.2f}'
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
        
        plt.tight_layout()
        plt.show()

    def report(self):
        # Print summary statistics with color
        print("\nTest Results Summary:")
        print(f"Total Predictions: {self.size}")
        print(f"{GREEN}Correct (Green): {self.green_count} ({self.green_count/self.size*100:.1f}%){RESET}")
        print(f"{ORANGE}Close (Orange): {self.orange_count} ({self.orange_count/self.size*100:.1f}%){RESET}")
        print(f"{RED}Wrong (Red): {self.red_count} ({self.red_count/self.size*100:.1f}%){RESET}")
        print(f"Average Error: ${self.average_error:,.2f}")
        print(f"RMSLE: {self.rmsle:.2f}")
        
        title = f"{self.title} Error=${self.average_error:,.2f}  RMSLE={self.rmsle:.2f}  HIT={self.green_count/self.size*100:.1f}%"
        self.chart(title)

    def run(self):
        # Progress bar for overall testing
        with tqdm(total=self.size, desc="Testing Progress", 
                 bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            
            # Progress bar for correct predictions
            expected_correct = int(self.size * 0.7)  # Set expected correct to 70%
            with tqdm(total=expected_correct, desc="Correct Predictions", 
                     bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}") as correct_pbar:
                
                for i in range(self.size):
                    color = self.run_datapoint(i)
                    pbar.update(1)
                    
                    if color == 'green' and correct_pbar.n < expected_correct:
                        correct_pbar.update(1)
        
        self.report()

    @classmethod
    def test(cls, function, data):
        cls(function, data).run()