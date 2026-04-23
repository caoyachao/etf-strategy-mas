"""
数据获取模块

负责从 baostock 拉取 ETF 历史 K 线数据，并提供批量加载接口。

外部入口：
    load_all_data(etf_pool, start_date, end_date) -> Dict[str, pd.DataFrame]
    get_benchmark_data(start_date, end_date) -> pd.DataFrame
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import baostock as bs

_BS_LOGGED_IN: bool = False


def _ensure_bs_login() -> None:
    """确保 baostock 已登录"""
    global _BS_LOGGED_IN
    if not _BS_LOGGED_IN:
        bs.login()
        _BS_LOGGED_IN = True


def _get_bs_code(symbol: str) -> str:
    """交易所代码转换"""
    prefix = symbol[:2]
    if prefix in ("51", "52", "53", "54", "56", "58"):
        return f"sh.{symbol}"
    elif prefix in ("15", "16", "18"):
        return f"sz.{symbol}"
    return f"sh.{symbol}"


def _validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """数据校验与清洗"""
    # 确保数值列是 float 类型
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 去除停牌日（全部价格为0）
    df = df[df["close"] > 0].copy()

    # 去除缺失值
    df = df.dropna(subset=["close"])

    return df


def get_etf_data(
    symbol: str,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    获取单只 ETF 的历史 K 线数据

    Args:
        symbol: ETF 代码，如 "510300"
        start_date: 开始日期，格式 "YYYY-MM-DD"
        end_date: 结束日期，格式 "YYYY-MM-DD"
        use_cache: 是否使用本地缓存

    Returns:
        DataFrame 含 date, open, high, low, close, volume
    """
    cache_dir = Path("data")
    cache_file = cache_dir / f"{symbol}_{start_date}_{end_date}.csv"

    # 优先使用缓存
    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file)
        if not df.empty:
            return df

    # 从 baostock 获取
    _ensure_bs_login()
    bs_code = _get_bs_code(symbol)

    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",  # 前复权
    )

    if rs.error_code != "0":
        raise RuntimeError(f"获取 {symbol} 数据失败: {rs.error_msg}")

    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())

    if not data_list:
        raise RuntimeError(f"{symbol} 在指定日期范围内无数据")

    df = pd.DataFrame(data_list, columns=["date", "open", "high", "low", "close", "volume"])
    df = _validate_data(df)

    # 保存缓存
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)

    return df


def get_benchmark_data(
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """获取沪深300ETF(510300)作为基准"""
    return get_etf_data("510300", start_date, end_date, use_cache)


def load_all_data(
    etf_pool: List[str],
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    统一加载所有 ETF 数据（含技术指标）。

    Returns:
        dict: {symbol: DataFrame(...)}，每列含 date, open, high, low, close, volume
    """
    from data.indicators import calculate_indicators

    data = {}
    for symbol in etf_pool:
        try:
            df = get_etf_data(symbol, start_date, end_date, use_cache)
            df = calculate_indicators(df)
            data[symbol] = df
        except Exception as e:
            print(f"[data.fetcher] 跳过 {symbol}（获取失败: {e}）", flush=True)
            continue

    if not data:
        raise RuntimeError("所有 ETF 数据获取失败，请检查网络或 baostock 可用性。")

    return data
