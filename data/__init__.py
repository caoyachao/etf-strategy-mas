"""
数据层模块

提供 ETF 数据获取、指标计算、基本面数据、数据摘要的统一接口。

主要入口：
    from data import load_all_data, get_fundamental_data, get_data_summary
"""

from data.fetcher import get_etf_data, get_benchmark_data, load_all_data
from data.indicators import calculate_indicators
from data.fundamentals import get_fundamental_data
from data.summary import get_data_summary, get_etf_summary_for_node

__all__ = [
    "get_etf_data",
    "get_benchmark_data",
    "load_all_data",
    "calculate_indicators",
    "get_fundamental_data",
    "get_data_summary",
    "get_etf_summary_for_node",
]
