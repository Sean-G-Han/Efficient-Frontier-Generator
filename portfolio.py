import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import scipy.optimize as sc
import plotly.graph_objects as go

class Portfolio:

    @classmethod
    def __getData(cls, startDate: datetime, endDate: datetime, stocks: list[str]) -> pd.Series:
        #--Get Data
        data = yf.download(stocks, start=startDate, end=endDate)["Close"]
        return data.pct_change()

    @classmethod
    def create(cls, startDate: datetime = None, endDate: datetime = None, stocks: list[str] = None,
               rf: float = 0, weights: list[int] = None, formattedData: pd.DataFrame = None) -> 'Portfolio':
        """Factory Method that creates a new portfolio"""
        #--Get Data
        if formattedData is None:
            data = Portfolio.__getData(startDate, endDate, stocks)
        else:
            data = formattedData
        data.reset_index(drop=True, inplace=True)
        data = data.rename_axis('Index')
        #--Get log meanReturns and covariance matrix
        returns = np.log(data + 1)
        meanReturns = returns.mean() * 252
        covMatrix = returns.cov() * 252
        #--Calculate weights
        if weights is None:
            weights = [1/len(meanReturns) for _ in meanReturns]
        else:
            weights = weights
        return Portfolio(meanReturns, covMatrix, rf, data, weights)
    
    def __init__(self, meanReturns: pd.Series, covMatrix: pd.DataFrame,
                 rf: float, data: pd.DataFrame, weights: list[float]) -> None:
        """Initializer"""
        #--Used to clone another portfolio
        self.data = data
        self.covMatrix = covMatrix
        self.meanReturns = meanReturns
        self.rf = rf
        self.weights = pd.Series(weights, index=meanReturns.index)
        #--Stats of Portfolio
        self.stdDev = self.__stdDev(weights)
        self.meanReturn = self.__meanReturn(weights)
        self.sharpeRatio = -self.__negativeSR(weights)

    #Calculate Stats of Porfolio
    def __negativeSR(self, weights: pd.Series = None) -> float:
        """Calculates Negative Sharpe Ratio of Portfolio with weights"""
        return (self.rf - self.__meanReturn(weights))/self.__stdDev(weights)
    
    def __stdDev(self, weights: pd.Series = None) -> float:
        """Calculates Standard Deviation of Portfolio with weights"""
        cov = np.exp(self.covMatrix) - 1
        return np.sqrt(np.dot(weights, np.dot(cov, weights)))
    
    def __meanReturn(self, weights: pd.Series = None) -> float:
        """Calculates Mean Returns of Portfolio with weights"""
        return np.exp(np.dot(weights, self.meanReturns)) - 1
    
    #Optimise Portfolio
    def __optimize(self, fun, constraints: dict,
                 constraintSet: tuple[float, float] = (0,1)) -> 'Portfolio':
        """Base optimiser function"""
        numAssets = len(self.weights)
        bounds = tuple(constraintSet for asset in range(numAssets))
        result = sc.minimize(fun, self.weights, 
                             method='SLSQP', bounds=bounds, constraints=constraints)
        return Portfolio(self.meanReturns, self.covMatrix, self.rf, self.data, result['x'])

    def maximizeSR(self: 'Portfolio',
                   constraintSet: tuple[float, float] = (0, 1)) -> 'Portfolio':
        """Finds a portfollio that has the highest possible sharpe ratio"""
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        return self.__optimize(self.__negativeSR, constraints, constraintSet)
    
    def minimizeVar(self: 'Portfolio',
                    constraintSet: tuple[float, float] = (0, 1)) -> 'Portfolio':
        """Finds a portfollio that has the lowest possible variance"""
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        return self.__optimize(self.__stdDev, constraints, constraintSet)
    
    def efficientVar(self: 'Portfolio', expectedReturns: float,
                     constraintSet: tuple[float, float] = (0, 1)) -> 'Portfolio':
        """Finds a portfollio that has the lowest variance for a given return"""
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'eq', 'fun': lambda x: self.__meanReturn(x) - expectedReturns})
        return self.__optimize(self.__stdDev, constraints, constraintSet)
    
    #Testing and Graphing
    def test(self, sampleOffset: tuple[int, int], testOffset: tuple[int, int], constraint: tuple[int, int] = (0,1), 
             fun = lambda x, constraint: x.minimizeVar(constraint)):
        """Test function"""
        maxID = self.data.index.max()
        sampleData = self.data.iloc[maxID - sampleOffset[0] : maxID - sampleOffset[1] + 1]
        testData = self.data.iloc[maxID - testOffset[0] : maxID - testOffset[1] + 1]
        sample = Portfolio.create(stocks=self.weights.index.tolist(), 
                                  formattedData=sampleData)
        optimizedSample = fun(sample, constraint)
        test = Portfolio.create(stocks=self.weights.index.tolist(), weights=optimizedSample.weights, 
                                formattedData=testData)
        return optimizedSample, test
    
    def efficientFrontier(self, steps: int = 30, constraint: tuple[int, int] = (0,1)) -> list['Portfolio']:
        minVarReturns = self.minimizeVar(constraint).meanReturn
        maxSRReturns = self.maximizeSR(constraint).meanReturn
        targetReturns = np.linspace(minVarReturns, maxSRReturns, steps)
        list = []
        for x in targetReturns:
            list.append(self.efficientVar(x, constraint))
        #--MaxSR
        maxSR = list[len(list) - 1]
        MaxSharpeRatio = go.Scatter(
            name='Maximium Sharpe Ratio',
            mode='markers',
            x=[round(maxSR.stdDev * 100, 2)],
            y=[round(maxSR.meanReturn * 100, 2)],
            marker=dict(color='red',size=14,line=dict(width=3, color='black')),
            hovertext=maxSR.getWeights()
        )

        #--MinVar
        minVar = list[0]
        MinVol = go.Scatter(
            name='Minimum Volatility',
            mode='markers',
            x=[round(minVar.stdDev * 100, 2)],
            y=[round(minVar.meanReturn * 100, 2)],
            marker=dict(color='green',size=14,line=dict(width=3, color='black')),
            hovertext=minVar.getWeights()
        )

        EF_curve = go.Scatter(
            name='Efficient Frontier',
            mode='lines',
            x=[round(portfolio.stdDev*100, 2) for portfolio in list],
            y=[round(target*100, 2) for target in targetReturns],
            line=dict(color='black', width=4, dash='dashdot'),
            hovertext=[portfolio.getWeights() for portfolio in list]
        )

        data = [MaxSharpeRatio, MinVol, EF_curve]

        layout = go.Layout(
        title = 'Portfolio Optimisation with the Efficient Frontier',
        yaxis = dict(title='Annualised Return (%)'),
        xaxis = dict(title='Annualised Volatility (%)'),
        showlegend = True,
        legend = dict(
            x = 0.75, y = 0, traceorder='normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2),
        width=1200,
        height=900)
    
        fig = go.Figure(data=data, layout=layout)
        fig.show()
        return list
    
    def getWeights(self):
        series = str(self.weights.map(lambda x: ": " + str(round(x * 100, 2)) + "%       "))
        return series
        
    def __str__(self) -> str:
        """Automatically converts portfolio to a string format"""
        return ("Std Deviation: " + str(round(self.stdDev*100, 3)) + "%\n" +
                "Mean Return: " + str(round(self.meanReturn*100, 3)) + "%\n" + 
                "Sharpe Ratio: " + str(round(self.sharpeRatio, 3)) + "\n"
                "Weights:\n" + str(self.weights.map(lambda x: str(round(x * 100, 2)) + "%")))