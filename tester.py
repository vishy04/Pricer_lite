import math
import matplotlib.pyplot as plt
import pandas as pd

class Tester:
    def __init__(self, data=None):
        """Initialize the Tester with data and empty result lists"""
        self.data = data
        self.predictor = None
        self.relative_errors = []
        self.percentage_errors = []
        self.mape_components = []
        self.smape_components = []
        self.log_cash_errors = []
        self.absolute_errors = []
        self.truths = []
        self.guesses = []
        
        # Color mapping for output
        self.COLOR_MAP = {
            "green": "\033[92m",
            "yellow": "\033[93m", 
            "red": "\033[91m"
        }
        self.RESET = "\033[0m"

    def test_function(self, predictor_function, data=None, num_samples=None):
        """
        Class method to test any function on the data
        
        Args:
            predictor_function: Any function that takes a datapoint and returns a price prediction
            data: Optional data to use (defaults to self.data)
            num_samples: Optional number of samples to test (defaults to all data)
        
        Returns:
            dict: Summary of test results
        """
        # Use provided data or default to instance data
        test_data = data if data is not None else self.data
        if test_data is None:
            raise ValueError("No data provided for testing")
        
        # Reset results for new test
        self._reset_results()
        
        # Set the predictor function
        self.predictor = predictor_function
        self.data = test_data
        
        # Determine how many samples to test
        total_samples = len(test_data)
        samples_to_test = min(num_samples or total_samples, total_samples)
        
        print(f"Testing function '{predictor_function.__name__}' on {samples_to_test} samples...")
        print("-" * 80)
        
        # Run tests
        error_count = 0
        for i in range(samples_to_test):
            try:
                self.run_datapoint(i)
            except Exception as e:
                print(f"Error on datapoint {i}: {str(e)}")
                error_count += 1
                continue
        
        if error_count > 0:
            print(f"\nWarning: {error_count} datapoints failed to process")
        
        # Generate report
        results = self._calculate_summary()
        self.report()
        
        return results
    
    def _reset_results(self):
        """Reset all result lists for a new test"""
        self.relative_errors = []
        self.percentage_errors = []
        self.mape_components = []
        self.smape_components = []
        self.log_cash_errors = []
        self.absolute_errors = []
        self.truths = []
        self.guesses = []
    
    def _calculate_summary(self):
        """Calculate and return summary statistics"""
        if not self.relative_errors:
            return {"error": "No valid results"}
        
        return {
            "total_samples": len(self.relative_errors),
            "avg_relative_error": sum(self.relative_errors) / len(self.relative_errors),
            "avg_percentage_error": sum(self.percentage_errors) / len(self.percentage_errors),
            "mape": sum(self.mape_components) / len(self.mape_components) * 100,
            "smape": sum(self.smape_components) / len(self.smape_components) * 100,
            "accuracy_within_10pct": sum(1 for e in self.relative_errors if e < 0.10) / len(self.relative_errors) * 100,
            "accuracy_within_20pct": sum(1 for e in self.relative_errors if e < 0.20) / len(self.relative_errors) * 100,
            "accuracy_within_30pct": sum(1 for e in self.relative_errors if e < 0.30) / len(self.relative_errors) * 100,
        }

    def run_datapoint(self, i):
        """Run prediction on a single datapoint and calculate metrics"""
        # Access DataFrame row correctly
        if isinstance(self.data, pd.DataFrame):
            datapoint = self.data.iloc[i]
            prompt = datapoint["prompt"]
            truth = datapoint["price"]
        else:
            # For list of dictionaries
            datapoint = self.data[i]
            prompt = datapoint.get("prompt", "Unknown item")
            truth = datapoint["price"]
        
        # Get prediction - this is where errors might occur
        guess = self.predictor(datapoint)
        
        # Only store values if we got this far without error
        self.truths.append(truth)
        self.guesses.append(guess)
        
        # Error metrics
        relative_error = abs(guess-truth)/truth 
        percentage_error = (relative_error)*100
        mape_component = abs(truth-guess) / truth
        smape_component = abs(truth-guess) / ((truth + guess)/2)
        log_cosh_error = math.log(math.cosh(guess-truth))
        absolute_error = abs(guess-truth)

        self.relative_errors.append(relative_error)
        self.percentage_errors.append(percentage_error)
        self.mape_components.append(mape_component)
        self.smape_components.append(smape_component)
        self.log_cash_errors.append(log_cosh_error)
        self.absolute_errors.append(absolute_error)

        color = self.color_for_relative_error(relative_error)
        title = self.extract_title(prompt)
        
        print(f"{self.COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} "
              f"Rel Err: {percentage_error:.1f}% "
              f"LCHE: {log_cosh_error:.3f} Item: {title}{self.RESET}")

    def extract_title(self, prompt):
        """Extract item title from prompt"""
        try:
            lines = prompt.split('\n')
            for line in lines:
                if line.strip() and 'How much does this cost' not in line and 'Price is' not in line:
                    return line.strip()[:50] + "..." if len(line.strip()) > 50 else line.strip()
            return "Unknown item"
        except:
            return "Unknown item"
        
    def color_for_relative_error(self, relative_error):
        """Determine color based on relative error"""
        if relative_error < 0.15:
            return "green"
        elif relative_error < 0.3:
            return "yellow"
        else:
            return "red"
        
    def report(self):
        """Enhanced reporting with multiple metrics"""
        if not self.absolute_errors:
            print("No results to report")
            return
        
        # Calculate all metrics
        mape = sum(self.mape_components) / len(self.mape_components) * 100
        smape = sum(self.smape_components) / len(self.smape_components) * 100
        
        # Accuracy within thresholds
        within_10_pct = sum(1 for e in self.relative_errors if e < 0.10)
        within_20_pct = sum(1 for e in self.relative_errors if e < 0.20)
        within_30_pct = sum(1 for e in self.relative_errors if e < 0.30)
        
        total = len(self.relative_errors)
        
        print(f"\n{'='*60}")
        print(f"TESTING RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total Samples: {total}")
        print(f"MAPE: {mape:.2f}%")
        print(f"SMAPE: {smape:.2f}%")
        print(f"Accuracy within 10%: {within_10_pct}/{total} ({within_10_pct/total*100:.1f}%)")
        print(f"Accuracy within 20%: {within_20_pct}/{total} ({within_20_pct/total*100:.1f}%)")
        print(f"Accuracy within 30%: {within_30_pct}/{total} ({within_30_pct/total*100:.1f}%)")

        # Create chart
        self.enhanced_chart()

    def enhanced_chart(self):
        """Create multiple charts for better analysis"""
        if not self.truths:
            print("No data to chart")
            return
        
        # Ensure all lists have the same length
        min_length = min(len(self.truths), len(self.guesses), len(self.relative_errors))
        if min_length == 0:
            print("No valid data to chart")
            return
            
        # Trim all lists to the same length in case of mismatches
        truths = self.truths[:min_length]
        guesses = self.guesses[:min_length]
        relative_errors = self.relative_errors[:min_length]
        percentage_errors = self.percentage_errors[:min_length]
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Predictions vs Truth
        max_val = max(max(truths), max(guesses))
        ax1.plot([0, max_val], [0, max_val], 'b--', alpha=0.7, label='Perfect Prediction')
        scatter_colors = ['green' if e < 0.2 else 'orange' if e < 0.4 else 'red' 
                         for e in relative_errors]
        ax1.scatter(truths, guesses, c=scatter_colors, alpha=0.6, s=20)
        ax1.set_xlabel('True Price ($)')
        ax1.set_ylabel('Predicted Price ($)')
        ax1.set_title('Predictions vs Ground Truth')
        ax1.legend()
        
        # 2. Error distribution
        ax2.hist(percentage_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=20, color='orange', linestyle='--', label='20% threshold')
        ax2.axvline(x=50, color='red', linestyle='--', label='50% threshold')
        ax2.set_xlabel('Percentage Error (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Percentage Errors')
        ax2.legend()

        # 3. Error vs True Price
        ax3.scatter(truths, percentage_errors, alpha=0.6, s=20)
        ax3.set_xlabel('True Price ($)')
        ax3.set_ylabel('Percentage Error (%)')
        ax3.set_title('Error vs Price Level')
        ax3.axhline(y=20, color='orange', linestyle='--', alpha=0.7)
        
        # 4. Cumulative accuracy
        sorted_errors = sorted(percentage_errors)
        cumulative_pct = [i/len(sorted_errors)*100 for i in range(1, len(sorted_errors)+1)]
        ax4.plot(sorted_errors, cumulative_pct, 'b-', linewidth=2)
        ax4.axvline(x=10, color='green', linestyle='--', label='10% error')
        ax4.axvline(x=20, color='orange', linestyle='--', label='20% error')
        ax4.axvline(x=30, color='red', linestyle='--', label='30% error')
        ax4.set_xlabel('Percentage Error (%)')
        ax4.set_ylabel('Cumulative Percentage of Predictions')
        ax4.set_title('Cumulative Error Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()