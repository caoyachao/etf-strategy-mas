"""
技术指标计算模块

计算 ETF 数据的常用技术指标：MA、RSI、MACD、布林带、波动率。

外部入口：
    calculate_indicators(df) -> pd.DataFrame
"""

import numpy as np
import pandas as pd


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算技术指标

    新增列:
    - ma5/20/60: 移动平均线
    - rsi: RSI(14)
    - macd, macd_signal, macd_hist: MACD
    - bb_upper, bb_lower, bb_mid, bb_pct: 布林带
    - volatility_20: 20日年化波动率
    """
    df = df.copy()
    close = df["close"]

    # MA
    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 布林带
    df["bb_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # 波动率（20日年化）
    df["volatility_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    return df
