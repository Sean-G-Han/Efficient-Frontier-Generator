import pandas as pd
from portfolio import *

endDate = pd.to_datetime("2024-07-05")
startDate = pd.to_datetime("2010-07-05")
stocks = ["WMT", "JPM", "BAC", "COST", "SPY", "QQQ"]
p = Portfolio.create(startDate, endDate, stocks)

#Back Test Model
numTest = 160 # backtesting for the past 160 months
predictReturns = []
actualReturns = []
for i in range(1, numTest):
    #Test the accuracy by taking the previous quarter result to estimate the next quarter"
    offset = i * 21 # Test for every month
    #test() is set to test the accuracy of min-variance. if you want to test max-Sharpe-Ratio, add "fun = lambda x, constraint: x.maximizeSR(constraint)"
    prediction, reality = p.test((126+ offset, 63 + offset), (63 + offset, 1 + offset))
    predictReturns.append(prediction.meanReturn)
    actualReturns.append(reality.meanReturn)

meanPredict = sum(predictReturns)/numTest
meanActual = sum(actualReturns)/numTest
print("\nPerformance:",
      "\nMean Expected returns (Annualized): ", str(meanPredict),
      "\nMean Actual returns   (Annualized): ", str(meanActual), "\n")

#Efficiency Frontier
#endDate stays the same
startDate = pd.to_datetime("2024-04-05")
x = Portfolio.create(startDate, endDate, stocks)
list = x.efficientFrontier() #creates efficient frontier in web broswer