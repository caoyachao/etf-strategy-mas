"""
基本面数据获取模块

从 baostock 获取 ETF 基本信息、净值、溢价率等。

外部入口：
    get_fundamental_data(etf_pool, date) -> Dict[str, Dict]
"""

from datetime import datetime
from typing import Dict, List, Optional

import baostock as bs

from data.fetcher import _ensure_bs_login, _get_bs_code


def get_fundamental_data(
    etf_pool: List[str],
    date: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    获取 ETF 基本面数据。

    目前从 baostock 的 query_etf_info / query_etf_nav 获取，
    若接口不可用则返回空 dict，不阻塞流程。

    Returns:
        {symbol: {"nav": float, "premium": float, "scale": float, ...}}
    """
    _ensure_bs_login()
    result = {}

    for symbol in etf_pool:
        info = {"symbol": symbol}

        # ETF 基本信息（净值、规模等）
        try:
            rs = bs.query_etf_info(code=symbol)
            if rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                info["fund_name"] = row[2] if len(row) > 2 else ""
                info["fund_manager"] = row[3] if len(row) > 3 else ""
                info["scale"] = float(row[5]) if len(row) > 5 else None
        except Exception:
            pass

        # ETF 净值（最新）
        try:
            today = date or datetime.now().strftime("%Y-%m-%d")
            rs = bs.query_etf_nav(code=symbol, date=today)
            if rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                info["nav"] = float(row[2]) if len(row) > 2 else None
        except Exception:
            pass

        # 溢价率 = (收盘价 - 净值) / 净值
        try:
            bs_code = _get_bs_code(symbol)
            rs = bs.query_latest_price(bs_code)
            if rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                close = float(row[4]) if len(row) > 4 else None
                if close and info.get("nav"):
                    info["premium"] = round((close - info["nav"]) / info["nav"], 4)
                info["latest_close"] = close
        except Exception:
            pass

        result[symbol] = info

    return result
