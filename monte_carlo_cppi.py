##############################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import IPython.display as ipi

##############################################################################################################

def monte_carlo_cppi_disp(number_of_scenarios = 50, mu = 0.07, sigma = 0.15, multiplier = 3, floor_rate = 0, 
                          riskfree_rate = 0.03, steps_per_year = 12, y_max = 100):
    """
    Plot the results of a Monte Carlo Simulation of CPPI:

    1- Use "geometric_brownian_motion" function to create set of returns as risky returns.

    2- Use "cppi_running" function to use the risky returns for CPPI strategy.

    3- You can change the parameters of interactive show.
    
    """

##############################################################################################################

    def geometric_brownian_motion(number_of_years = 10, number_of_scenarios = 1000, mu = 0.07, 
                                  sigma = 0.15, steps_per_year = 12, initial_value = 100, prices = True):
        """
        Geometric Brownian Motion trajectories through Monte Carlo:
        
        1- "number_of_years" is about the number of years to generate data for.

        2- "mu" is annualized drift on market return.

        3- "sigma" is annualized volatility on market return.

        4- "steps_per_year" is granularity of the simulation for a year's steps.
        
        5- "initial_value" is initial value for amount of investing.

        """
        dt = 1 / steps_per_year
        number_of_steps = int(number_of_years * steps_per_year) + 1
        returns = np.random.normal(loc = (1 + mu) ** dt, scale = (sigma * np.sqrt(dt)), 
                                size = (number_of_steps, number_of_scenarios))
        returns[0] = 1
        returns_values = initial_value * pd.DataFrame(returns).cumprod() if prices else returns - 1
        return returns_values


    def cppi_running(risky_returns, safe_returns = None, multiplier = 3, initial_value = 1000,
                     floor_rate = 0.8, riskfree_rate = 0.03, drawdown = None):
        """
        Run a backtest of the CPPI strategy:

        1- CPPI stands for constant proportion portfolio insurance.
        
        2- Adjust a set of returns for the risky asset.

        3- You can change the values of "cppi_running" function arguments.

        """

        dates = risky_returns.index
        number_of_steps = len(dates)
        account_value = initial_value
        floor_value = initial_value * floor_rate
        peak = account_value

        if isinstance(risky_returns, pd.Series): 
            risky_returns = pd.DataFrame(risky_returns, columns = ["R"])

        if safe_returns is None:
            safe_returns = pd.DataFrame().reindex_like(risky_returns)
            safe_returns.values[:] = riskfree_rate / 12

        account_history = pd.DataFrame().reindex_like(risky_returns)
        risky_weight_history = pd.DataFrame().reindex_like(risky_returns)
        cushion_history = pd.DataFrame().reindex_like(risky_returns)
        floorval_history = pd.DataFrame().reindex_like(risky_returns)
        peak_history = pd.DataFrame().reindex_like(risky_returns)

        for step in range(number_of_steps):

            if drawdown is not None:
                peak = np.maximum(peak, account_value)
                floor_value = peak * (1 - drawdown)

            cushion = (account_value - floor_value) / account_value
            risky_weight = multiplier * cushion
            risky_weight = np.minimum(risky_weight, 1)
            risky_weight = np.maximum(risky_weight, 0)
            safe_weight = 1 - risky_weight
            risky_allocation = account_value * risky_weight
            safe_allocation = account_value * safe_weight
            account_value = risky_allocation * (1 + risky_returns.iloc[step]) + safe_allocation * (1 + safe_returns.iloc[step])
            cushion_history.iloc[step] = cushion
            risky_weight_history.iloc[step] = risky_weight
            account_history.iloc[step] = account_value
            floorval_history.iloc[step] = floor_value
            peak_history.iloc[step] = peak
        
        risky_wealth = initial_value * (1 + risky_returns).cumprod()
        
        backtest_result = {"Wealth ": account_history,
                           "Risky Wealth ": risky_wealth, 
                           "Risk Budget ": cushion_history,
                           "Risky Allocation ": risky_weight_history,
                           "Multiplier ": multiplier,
                           "initial_value Value ": initial_value,
                           "Floor Value Rate ": floor_rate,
                           "Risky_returns ":risky_returns,
                           "Safe_returns ": safe_returns,
                           "Drawdown ": drawdown,
                           "Peak History ": peak_history,
                           "Floor History ": floorval_history}
        return backtest_result    

##############################################################################################################   

    initial_value = 100
    simulation_returns = geometric_brownian_motion(number_of_scenarios = number_of_scenarios, mu = mu, sigma = sigma, 
                                                   prices = False, steps_per_year = steps_per_year)
    risky_returns = pd.DataFrame(simulation_returns)
    backtest_running = cppi_running(risky_returns = pd.DataFrame(risky_returns), riskfree_rate = riskfree_rate,
                                    multiplier = multiplier, initial_value = initial_value, floor_rate = floor_rate)
    wealth = backtest_running["Wealth "]
    y_max = wealth.values.max() * y_max / 100
    terminal_wealth = wealth.iloc[-1]
    terminal_wealth_mean = terminal_wealth.mean()
    terminal_wealth_median = terminal_wealth.median()
    failures = np.less(terminal_wealth, initial_value * floor_rate)
    number_of_failures = failures.sum()
    failure_percent = number_of_failures / number_of_scenarios
    short_fall = np.dot(terminal_wealth - initial_value * floor_rate, failures) / number_of_failures if number_of_failures > 0 else 0.0

    fig, (wealth_ax, hist_ax) = plt.subplots(nrows = 1, ncols = 2, sharey = True, gridspec_kw = {'width_ratios':[3,2]}, figsize = (24, 9))
    plt.subplots_adjust(wspace = 0.0)
    
    wealth.plot(ax = wealth_ax, legend = False, alpha = 0.3, color = "indianred")
    wealth_ax.axhline(y = initial_value, ls = ":", color = "black")
    wealth_ax.axhline(y = initial_value * floor_rate, ls = "--", color = "red")
    wealth_ax.set_ylim(top = y_max)
    
    terminal_wealth.plot.hist(ax = hist_ax, bins = 50, ec = 'w', fc = 'indianred', orientation = 'horizontal')
    hist_ax.axhline(y = initial_value, ls = ":", color = "black")
    hist_ax.axhline(y = terminal_wealth_mean, ls = ":", color = "blue")
    hist_ax.axhline(y = terminal_wealth_median, ls = ":", color = "purple")
    hist_ax.annotate(f"Mean: ${int(terminal_wealth_mean)}", xy = (0.7, 0.9), xycoords = 'axes fraction', fontsize = 24)
    hist_ax.annotate(f"Median: ${int(terminal_wealth_median)}", xy = (0.7, 0.85), xycoords = 'axes fraction', fontsize = 24)
    if (floor_rate > 0.01):
        hist_ax.axhline(y = initial_value * floor_rate, ls = "--", color = "red", linewidth = 3)
        hist_ax.annotate(f"Violations: {number_of_failures} ({failure_percent * 100:2.2f}%)\nE(shortfall)=${short_fall:2.2f}", xy = (0.7, 0.7), xycoords = 'axes fraction', fontsize = 24)

cppi_controls = widgets.interactive(monte_carlo_cppi_disp,
                                    number_of_scenarios = widgets.IntSlider(min = 1, max = 1000, step = 5, value = 50), 
                                    mu = (0, 0.2, 0.01),
                                    sigma = (0, 0.3, 0.05),
                                    floor_rate = (0, 2, 0.1),
                                    multiplier = (1, 5, 0.5),
                                    riskfree_rate = (0, 0.05, 0.01),
                                    steps_per_year = widgets.IntSlider(min = 1, max = 12, step = 1, value = 12,
                                                                       description = "Time Periods/Year"),
                                    y_max = widgets.IntSlider(min = 0, max = 100, step = 1, value = 100,
                                                              description = "Zoom on Y Axis"))
ipi.display(cppi_controls)