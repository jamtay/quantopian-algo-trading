# Quality companies in an uptrend trading algorithm using quantopian
import quantopian.algorithm as algo

# import things need to run pipeline
from quantopian.pipeline import Pipeline

# import any built-in factors and filters being used
from quantopian.pipeline.filters import Q500US
from quantopian.pipeline.factors import Returns

# import any needed datasets
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data.morningstar import Fundamentals as ms
from quantopian.pipeline import  CustomFactor

# import optimize for trade execution
import quantopian.optimize as opt
import numpy as np
import pandas as pd

def initialize(context):

    # Set target number of securities to hold
    context.TARGET_SECURITIES = 5
    # Set the top number of quality companies to look at momentum for
    context.TOP_ROE_QTY = 100

    # Used for the trend following filter
    context.SPY = symbol('SPY')
    context.TF_SLOW_LOOKBACK = 100
    context.TF_FAST_LOOKBACK = 10

    # Set how many days to lookback over to determine momentum of a stock
    context.MOMENTUM_LOOKBACK_DAYS = 140
    # Initialize variables before being used
    context.stock_weights = pd.Series()

    # Set slippage to 0 so the algorithm doesn't try to avoid it. This would make it harder to see whats been bought and sold each month and in what quantity
    # Also when dealing with smaller amounts this is not as important
    set_slippage(slippage.FixedSlippage(spread = 0.0))

    # Create and attach pipeline for fetching all data
    algo.attach_pipeline(make_pipeline(context), 'pipeline')
    # Schedule functions to run once a month, 7 days before month end
    # Separate the stock selection from the execution for readability
    schedule_function(
        select_stocks_and_set_weights,
        date_rules.month_end(days_offset = 7),
        time_rules.market_open()
    )
    schedule_function(
        trade,
        date_rules.month_end(days_offset = 7),
        time_rules.market_open()
    )


def make_pipeline(context):
    # Use a universe of just the top 500 US stocks
    universe = Q500US()

    # Get simple factors to determine "quality" companies
    cash_return_latest = ms.cash_return.latest
    fcf_yield_latest = ms.fcf_yield.latest
    roic_latest = ms.roic.latest
    rev_growth_latest = ms.revenue_growth.latest

    cash_return = cash_return_latest.rank(mask=universe)
    fcf_yield = fcf_yield_latest.rank(mask=universe)
    roic = roic_latest.rank(mask=universe)

    rev_growth = rev_growth_latest.rank(mask=universe)
    value = (cash_return + fcf_yield).rank(mask=universe)
    # Combine factors to create one single ranking system
    quality = value + roic + rev_growth

    # Create a 'momentum' factor by looking back over the last 140 days (20 weeks)
    momentum = Returns(window_length=context.MOMENTUM_LOOKBACK_DAYS)

    # Filters for top quality and momentum to use in the selection criteria
    top_quality = quality.top(context.TOP_ROE_QTY, mask=universe) & cash_return_latest.notnull() & fcf_yield_latest.notnull() & roic_latest.notnull() & rev_growth_latest.notnull()
    top_quality_momentum = momentum.top(context.TARGET_SECURITIES, mask=top_quality)

    # Only return values to be used in the selection criteria by using the screen parameter
    pipe = Pipeline(columns={
                        'top_quality_momentum': top_quality_momentum,
                        },
                    screen=top_quality_momentum
                   )
    return pipe

def select_stocks_and_set_weights(context, data):

    # Get pipeline output and select stocks
    df = algo.pipeline_output('pipeline')

    # Define the rule to open/hold positions
    rule = 'top_quality_momentum'

    stocks_to_hold = df.query(rule).index
    # Set desired stock weights by equally weighting the stocks
    stock_weight = 1.0 / context.TARGET_SECURITIES
    context.stock_weights = pd.Series(index=stocks_to_hold, data=stock_weight)

def trade(context, data):

    # Create a TargetWeights objective
    target_weights = opt.TargetWeights(context.stock_weights)

    # Execute the order_optimal_portfolio method with above objective and any constraint
    # Add opt.MaxGrossExposure(1.0) as a constraint to insure not leverage is used
    constraints = []
    constraints.append(opt.MaxGrossExposure(1.0))
    order_optimal_portfolio(
        objective = target_weights,
        constraints = constraints
        )
