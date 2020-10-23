# coding=utf-8
import pandas as pd
import numpy as np
import empyrical as ep

from pyfolio import *
from pyfolio.utils import APPROX_BDAYS_PER_MONTH

def get_statistics(returns, benchmark_rets, positions):
    """
    获取衡量投资组合回测表现的统计指标，包括：
    Annual return         Cumulative return
    Annual volatility     Sharpe ratio
    Calmar ratio          Stability
    Max drawdown          Omega ratio
    Sortino ratio         Skew
    Kurtosis              Tail ratio 
    Daily value at risk   Gross leverage
    Alpha                 Beta
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    pd.Series
        衡量投资组合回测表现的统计指标序列
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats


def get_annual_return(returns, benchmark_rets, positions):
    """
    获取投资组合的年化收益率
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    float
        Annual return   
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats['Annual return']


def get_cumulative_return(returns, benchmark_rets, positions):
    """
    获取投资组合的累计收益率
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    float
        Cumulative return   
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats['Cumulative returns']


def get_annual_volatility(returns, benchmark_rets, positions):
    """
    获取投资组合的年化波动率
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    float
        Annual volatility  
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats['Annual volatility']


def get_sharpe_ratio(returns, benchmark_rets, positions):
    """
    获取投资组合的夏普率
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    float
        Sharpe ratio  
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats['Sharpe ratio']


def get_calmar_ratio(returns, benchmark_rets, positions):
    """
    获取投资组合的Calmar比率
    Calmar比率=年化收益率/最大回撤，描述收益与最大回撤之间的关系，值越大说明基金的表现越好
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    float
        Calmar ratio  
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats['Calmar ratio']


def get_stability(returns, benchmark_rets, positions):
    """
    获取投资组合累计对数收益率线性拟合的R-squared
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    float
        Stability  
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats['Stability']


def get_max_drawdown(returns, benchmark_rets, positions):
    """
    获取投资组合的最大回撤
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    float
        Max drawdown  
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats['Max drawdown']


def get_omega_ratio(returns, benchmark_rets, positions):
    """
    获取投资组合的Omega比率，值越大代表投资效益越好
    Omega比率利用了收益率分布的所有信息，考虑了所有的高阶矩，刻画了收益率风险的所有特征。
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    float
        Omega ratio
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats['Omega ratio']


def get_sortino_ratio(returns, benchmark_rets, positions):
    """
    获取投资组合的Sortino比率，值越大表明基金承担相同单位下行风险能获得更高的超额回报率。
    索提诺比率可以看做是夏普比率在衡量对冲基金/私募基金时的一种修正方式。
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    float
        Sortino ratio
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats['Sortino ratio']


def get_skew(returns, benchmark_rets, positions):
    """
    获取投资组合收益率的偏度。
    
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    float
        Skew
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats['Skew']


def get_kurtosis(returns, benchmark_rets, positions):
    """
    获取投资组合收益率的峰度。
    
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    float
        Kurtosis
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats['Kurtosis']


def get_tail_ratio(returns, benchmark_rets, positions):
    """
    获取投资组合收益率序列右尾（95%）与左尾（5%）的比值。
    
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    float
        Tail ratio
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats['Tail ratio']


def get_VaR(returns, benchmark_rets, positions):
    """
    获取投资组合收益率的在险价值。
    
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    float
        Daily value at risk
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats['Daily value at risk']



def get_gross_leverage(returns, benchmark_rets, positions):
    """
    获取投资组合总杠杆的平均值。
    
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    float
        Gross leverage
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats['Gross leverage']


def get_alpha(returns, benchmark_rets, positions):
    """
    获取投资组合的Alpha。
    
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    float
        Alpha
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats['Alpha']


def get_beta(returns, benchmark_rets, positions):
    """
    获取投资组合的Beta。
    
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    benchmark_rets : pd.Series
              基准组合收益率的时间序列
    positions : pd.DataFrame
              投资组合的持仓
              
    Returns
    ----------
    float
        Beta
    """
    stats=timeseries.perf_stats(returns, 
                                factor_returns=benchmark_rets, 
                                positions=positions,
                                transactions=None, 
                                turnover_denom='AGB')
    return stats['Beta']


def get_drawdowns(returns, top=10):
    """
    获取投资组合的前top个最大回撤的相关信息，包括如下字段：
    "Net drawdown in %"、“Peak date”、“Valley date”、“Recovery date”、“Duration”
    
    
    Parameters
    ----------
    returns : pd.Series
              投资组合收益率的时间序列
    top : int, optional
              最大回撤的数目
              
    Returns
    ----------
    pd.DataFrame
        df_drawdowns
    """
    df_drawdowns=timeseries.gen_drawdown_table(returns, top=top)
    return df_drawdowns


def get_cum_rets(returns):
    """
    获取投资组合的累计收益率序列
    
    Parameters
    ----------
    returns : pd.Series
              投资组合原始收益率序列
              
    Returns
    ----------
    pd.Series
        cum_rets
    """
    cum_rets = ep.cum_returns(returns, 1.0)
    return cum_rets


def get_vol_adj_cum_rets(returns, benchmark_rets):
    """
    获取投资组合经过基准收益波动率调整后的累计收益率序列，这样做得好处在于
    可以比较拥有不同波动率的投资组合的累计收益率
    
    Parameters
    ----------
    returns : pd.Series
              投资组合原始收益率序列
              
    benchmark_rets : pd.Series
              基准组合原始收益率序列
              
    Returns
    ----------
    pd.Series
        cum_rets_adj
    """
    bmark_vol = benchmark_rets.loc[returns.index].std()
    returns_adj = (returns / returns.std()) * bmark_vol
    cum_rets_adj = ep.cum_returns(returns_adj, 1.0)
    return cum_rets_adj


def get_cum_benchmark_rets(benchmark_rets):
    """
    获取投资组合的累计收益率序列
    
    Parameters
    ----------
    benchmark_rets : pd.Series
              基准组合原始收益率序列
              
    Returns
    ----------
    pd.Series
        cum_benchmark_rets
    """
    cum_benchmark_rets = ep.cum_returns(benchmark_rets, 1.0)
    return cum_benchmark_rets


def get_rolling_beta(returns, benchmark_rets, window=6):
    """
    获取投资组合的滚动beta，窗口期可以指定。
    
    Parameters
    ----------
    returns : pd.Series
              投资组合原始收益率序列
    
    benchmark_rets : pd.Series
              基准组合原始收益率序列
              
    window : int, optional
              窗口包含月份的数量
              
    Returns
    ----------
    pd.Series
        rolling_beta
    """
    rolling_beta = timeseries.rolling_beta(returns, 
                                           benchmark_rets, 
                                           rolling_window=APPROX_BDAYS_PER_MONTH * window)
    return rolling_beta


def get_rolling_volatility(returns, window=6):
    """
    获取收益序列的滚动波动率，窗口期可以指定。
    
    Parameters
    ----------
    returns : pd.Series
              收益率序列
              
    window : int, optional
              窗口包含月份的数量
              
    Returns
    ----------
    pd.Series
        rolling_vol
    """
    rolling_vol = timeseries.rolling_volatility(returns, 
                                                rolling_window = APPROX_BDAYS_PER_MONTH*window)
    return rolling_vol



def get_rolling_sharpe(returns, window=6):
    """
    获取收益序列的滚动夏普率，窗口期可以指定。
    
    Parameters
    ----------
    returns : pd.Series
              收益率序列
              
    window : int, optional
              窗口包含月份的数量
              
    Returns
    ----------
    pd.Series
        rolling_sharpe
    """
    rolling_sharpe = timeseries.rolling_sharpe(returns, 
                                               rolling_window = APPROX_BDAYS_PER_MONTH*window)
    return rolling_sharpe



def get_period_rets(returns, period='monthly'):
    """
    将原始的日度收益率序列压缩成指定周期的收益率序列，周期可选参数包括：周度、月度、年度
    
    Parameters
    ----------
    returns : pd.Series
              原始日度收益率序列
              
    period : str, optional
              周期("weekly","monthly","yearly")
              
    Returns
    ----------
    pd.Series or pd.DataFrame
        period_rets
    """
    if period=="weekly":
        weekly_ret_table = ep.aggregate_returns(returns, 'weekly')
        weekly_ret_table = weekly_ret_table.unstack().round(3)
        return weekly_ret_table
    elif period=="monthly":
        monthly_ret_table = ep.aggregate_returns(returns, 'monthly')
        monthly_ret_table = monthly_ret_table.unstack().round(3) 
        return monthly_ret_table
    else:
        ann_ret_series = pd.DataFrame(ep.aggregate_returns(returns,'yearly'),
                                      columns=['Annual Return'])
        return ann_ret_series


def get_event_rets(returns):
    """
    获取收益率序列处于压力事件覆盖时间区段的部分。
    
    Parameters
    ----------
    returns : pd.Series
              收益率序列
                            
    Returns
    ----------
    pd.DataFrame
        event_rets
    """
    event_rets = timeseries.extract_interesting_date_ranges(returns)
    return pd.DataFrame(event_rets)


def get_long_exposure(postions):
    """
    获取多头持仓（long exposure）占比的时间序列。
    
    Parameters
    ----------
    postions : pd.DataFrame
        投资组合的持仓矩阵
                            
    Returns
    ----------
    pd.Series
        long_exposure
    """
    pos_no_cash = positions.drop('cash', axis=1)
    long_exposure = pos_no_cash[pos_no_cash > 0].sum(axis=1) / positions.sum(axis=1)
    long_exposure.name="多头持仓占比"
    return long_exposure


def get_short_exposure(postions):
    """
    获取空头持仓（short exposure）占比的时间序列。
    
    Parameters
    ----------
    postions : pd.DataFrame
        投资组合的持仓矩阵
                            
    Returns
    ----------
    pd.Series
        short_exposure
    """
    pos_no_cash = positions.drop('cash', axis=1)
    short_exposure = pos_no_cash[pos_no_cash < 0].sum(axis=1) / positions.sum(axis=1)
    short_exposure.name="空头持仓占比"
    return short_exposure


def get_net_exposure(postions):
    """
    获取净持仓（net exposure）占比的时间序列。
    
    Parameters
    ----------
    postions : pd.DataFrame
        投资组合的持仓矩阵
                            
    Returns
    ----------
    pd.Series
        net_exposure
    """
    pos_no_cash = positions.drop('cash', axis=1)
    net_exposure = pos_no_cash.sum(axis=1) / positions.sum(axis=1)
    net_exposure.name="净持仓占比"
    return net_exposure


def get_top_exposure_shares(postions):
    """
    分别获取投资组合多头、空头及绝对持仓top10的股票
    
    Parameters
    ----------
    postions : pd.DataFrame
        投资组合的持仓矩阵
                            
    Returns
    ----------
    tuple
        top_long_shares, top_short_shares, top_abs_shares
    """
    positions_alloc = positions.divide(positions.sum(axis='columns'),axis='rows')
    top_long_shares, top_short_shares, top_abs_shares = pos.get_top_long_short_abs(positions_alloc)
    return top_long_shares, top_short_shares, top_abs_shares


def get_max_median_position_concentration(postions):
    """
    获取每个时间切片上最大的多头持仓比例、最小的空头持仓比例以及多头持仓比例和空头持仓比例的中位数
    
    Parameters
    ----------
    postions : pd.DataFrame
        投资组合的持仓矩阵
                            
    Returns
    ----------
    pd.DataFrame
        alloc_summary
    """
    alloc_summary = pos.get_max_median_position_concentration(positions)
    return alloc_summary


def get_holdings(postions):
    """
    获取投资组合建仓的股票数量、月度平均建仓股票数量、整个回测区间内的平均建仓股票数量
    
    Parameters
    ----------
    postions : pd.DataFrame
        投资组合的持仓矩阵
                            
    Returns
    ----------
    tuple: pd.Series, pd.Series, float
        daily_holdings, avg_holdings_by_month, avg_holdings_by_day
    """
    positions_alloc = positions.divide(positions.sum(axis='columns'),axis='rows')
    positions_alloc_no_cash = positions_alloc.copy().drop('cash', axis='columns')
    daily_holdings = positions_alloc_no_cash.replace(0, np.nan).count(axis=1)
    avg_holdings_by_month = daily_holdings.resample('1M').mean()  ##频率转换
    avg_holdings_by_day=daily_holdings.values.mean()
    return daily_holdings, avg_holdings_by_month, avg_holdings_by_day


def get_long_short_holdings(postions):
    """
    获取投资组合做多及卖空股票数目的时间序列。
    
    Parameters
    ----------
    postions : pd.DataFrame
        投资组合的持仓矩阵
                            
    Returns
    ----------
    tuple: pd.Series, pd.Series
        long_daily_holdings, short_daily_holdings
    """
    positions_alloc = positions.divide(positions.sum(axis='columns'),axis='rows')
    positions_alloc_no_cash = positions_alloc.copy().drop('cash', axis='columns')
    temp_position = positions_alloc_no_cash.replace(0, np.nan)
    long_daily_holdings = temp_position[temp_position > 0].count(axis=1)
    short_daily_holdings = temp_position[temp_position < 0].count(axis=1)
    long_daily_holdings.name="做多股票数目"
    short_daily_holdings.name="做空股票数目"
    return long_daily_holdings, short_daily_holdings


def get_gross_leverage_ts(postions):
    """
    获取投资组合总杠杆的时间序列。
    投资组合的总杠杆=投资组合每一期在个股上头寸绝对值之和/投资组合每一期在个股上的头寸与资金账户上余额之和
      
    Parameters
    ----------
    postions : pd.DataFrame
        投资组合的持仓矩阵
                            
    Returns
    ----------
    pd.Series
        gross_leverage_ts
    """
    gross_leverage_ts = timeseries.gross_lev(positions)
    return gross_leverage_ts


def get_sector_exposures(postions,sector_mappings):
    """
    获取投资组合在各行业上的净持仓金额，包含资金账户。
      
    Parameters
    ----------
    postions : pd.DataFrame
        投资组合的持仓矩阵
        
    sector_mappings : dict
        股票行业归属字典
                            
    Returns
    ----------
    pd.DataFrame
        sector_exposures
    """
    sector_exposures = pos.get_sector_exposures(positions,sector_mappings)
    return sector_exposures


def get_sector_alloc(postions,sector_mappings):
    """
    获取投资组合在各行业上的净持仓比例，包含资金账户。
      
    Parameters
    ----------
    postions : pd.DataFrame
        投资组合的持仓矩阵
        
    sector_mappings : dict
        股票行业归属字典
                            
    Returns
    ----------
    pd.DataFrame
        sector_alloc
    """
    sector_exposures = pos.get_sector_exposures(positions,sector_mappings)
    sector_alloc = pos.get_percent_alloc(sector_exposures)
    return sector_alloc

    
    
    

    
    

        
    


        
        
    







    