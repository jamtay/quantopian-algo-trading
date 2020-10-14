# Quality companies in an uptrend trading algorithm using quantopian
import quantopian.algorithm as algo

# import things need to run pipeline
from quantopian.pipeline import Pipeline

# import any built-in factors and filters being used
from quantopian.pipeline.filters import Q500US
from quantopian.pipeline.factors import SimpleMovingAverage as SMA
from quantopian.pipeline.factors import Returns

# import any needed datasets
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data.morningstar import Fundamentals as ms

# import optimize for trade execution
import quantopian.optimize as opt
import numpy as np
import pandas as pd

def initialize(context):
    # List of bond ETFs when market is trending downwards
    context.BONDS = [symbol('IEF'), symbol('TLT')]

    # Set target number of securities to hold
    context.TARGET_SECURITIES = 5
    # Set the top number of quality companies to look at momentum for
    context.TOP_ROE_QTY = 50

    # Used for the trend following filter
    context.SPY = symbol('SPY')
    context.TF_SLOW_LOOKBACK = 100
    context.TF_FAST_LOOKBACK = 10

    # Set how many days to lookback over to determine momentum of a stock
    context.MOMENTUM_LOOKBACK_DAYS = 140
    # Initialize variables before being used
    context.stock_weights = pd.Series()
    context.bond_weights = pd.Series()

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

    # Determine if the market is trending up or down. If the market is trending down then just buy bonds. If it is trending up then buy stocks
    # This is not an optimal solution and will obviously not work in all future market crashes
    spy_ma_fast_slice = SMA(inputs=[USEquityPricing.close],
                         window_length=context.TF_FAST_LOOKBACK)[context.SPY]
    spy_ma_slow_slice = SMA(inputs=[USEquityPricing.close],
                          window_length=context.TF_SLOW_LOOKBACK)[context.SPY]
    spy_ma_fast = SMA(inputs=[spy_ma_fast_slice], window_length=1)
    spy_ma_slow = SMA(inputs=[spy_ma_slow_slice], window_length=1)
    # If the 100 moving average crosses the 10 then alter the trend up variable
    trend_up = spy_ma_fast > spy_ma_slow

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
                        'trend_up': trend_up,
                        'top_quality_momentum': top_quality_momentum,
                        },
                    screen=top_quality_momentum
                   )
    return pipe

def select_stocks_and_set_weights(context, data):
    # Get pipeline output and select stocks
    df = algo.pipeline_output('pipeline')
    current_holdings = context.portfolio.positions

    # Define the rule to open/hold positions
    # top momentum and don't open in a downturn but, if held, then keep
    rule = 'top_quality_momentum & (trend_up or (not trend_up & index in @current_holdings))'
    stocks_to_hold = df.query(rule).index
    # Set desired stock weights by equally weighting the stocks
    stock_weight = 1.0 / context.TARGET_SECURITIES
    context.stock_weights = pd.Series(index=stocks_to_hold, data=stock_weight)
    # Set desired bond weight
    bond_weight = max(1.0 - context.stock_weights.sum(), 0) / len(context.BONDS)
    context.bond_weights = pd.Series(index=context.BONDS, data=bond_weight)

def trade(context, data):

    # Create a single series from the stock and bond weights
    total_weights = pd.concat([context.stock_weights, context.bond_weights])

    # Create a TargetWeights objective
    target_weights = opt.TargetWeights(total_weights)

    # Execute the order_optimal_portfolio method with above objective and any constraint
    # Add opt.MaxGrossExposure(1.0) as a constraint to insure not leverage is used
    constraints = []
    constraints.append(opt.MaxGrossExposure(1.0))
    order_optimal_portfolio(
        objective = target_weights,
        constraints = constraints
        )
    # Record the weights for insight into stock/bond mix and impact of trend following
    record(stocks=context.stock_weights.sum(), bonds=context.bond_weights.sum())
