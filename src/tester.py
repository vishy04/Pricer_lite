import math
import matplotlib.pyplot as plt
import pandas as pd


# COLOR MAP
GREEN = "\033[92m"
ORANGE = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red":RED, "orange": ORANGE, "green": GREEN}
class Tester :

    def __init__(self,predictor ,  data , title=None , size = 250):
        self.predictor = predictor 
        self.data = data 
        self.title = title or predictor.__name__.replace("_"," ").title()
        self.size = size 
        self.guesses = []
        self.truths = []
        self.errors = []
        self.lche = []
        self.sles= []
        self.colors = []

    def run_datapoint(self , i):
        datapoint = self.data.iloc[i]

        guess = float(self.predictor(datapoint)) #predicted output
        truth = float(datapoint['price']) #always be positive

        error = abs(truth - guess)
        log_error = math.log(truth+1) - math.log(guess+1)
        sle = log_error ** 2
        log_cosh_error = self.safe_log_cosh(error)

        color = self.color_for(error , truth) #for better outputs
        title = datapoint['title'] if len(datapoint['title']) <= 40 else datapoint['title'][:40] + '...'
        
        self.guesses.append(guess)
        self.truths.append(truth)

        self.errors.append(error)
        self.colors.append(color)
        self.lche.append(log_cosh_error)
        self.sles.append(sle)

        print(f"{COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}")
    
    def safe_log_cosh(self,x):
        """avoids overflow"""
        x = max(min(x, 500), -500)  # Cap between -500 and 500
        return math.log(math.cosh(x))

    def color_for(self , error , truth):
        if error < 40 or error/truth < 0.2:
            return 'green'
        elif error < 80 or error/truth < 0.4:
            return 'orange'
        else :
            return 'red'
        
    def chart(self,title):
        max_error = max(self.errors)
        plt.figure(figsize=(12,8))
        max_val = max(max(self.truths),max(self.guesses))
        plt.plot([0,max_val],[0,max_val],color='skyblue' , lw=2 , alpha=0.6)
        plt.scatter(self.truths,self.guesses,s=3,c=self.colors)
        plt.xlabel('True Values')
        plt.ylabel('Guess Values by Model')
        plt.xlim(0,max_val)
        plt.ylim(0,max_val)
        plt.title(title)
        plt.show()
    
    def report(self):
        average_error = sum(self.errors) / self.size 
        rmsle = math.sqrt(sum(self.sles)/self.size)
        HIT = sum(1 for color in self.colors if color =='green')
        title = f"{self.title} Error=${average_error:,.2f}  RMSLE={rmsle:,.2f}  HIT={HIT/self.size*100:.1f}%"
        self.chart(title)
    
    def run (self):
        self.error = 0 
        for i in range (self.size):
            self.run_datapoint(i)
        self.report()
    
    @classmethod
    def test(cls , function,data):
        cls(function,data).run()
