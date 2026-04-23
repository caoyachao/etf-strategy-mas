"""
数据摘要生成模块

为 LLM 分析节点生成可读的数据摘要文本。

外部入口：
    get_data_summary(data, fundamental, n_recent) -> str
    get_etf_summary_for_node(data, node_type, n_recent) -> str
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def get_data_summary(
    data: Dict[str, pd.DataFrame],
    fundamental: Dict[str, Dict],
    n_recent: int = 5,
) -> str:
    """
    生成所有 ETF 的近期数据摘要文本，供 LLM prompt 使用。

    输出格式：
      510300 (沪深300ETF)
        最新收盘价: 3.85 | 20日均: 3.82 | 60日均: 3.78
        RSI(14): 72.5 | MACD: 0.03(金叉) | 布林带: 80%
        净值: 3.84 | 溢价率: 0.26%
        近5日走势: 3.80 → 3.82 → 3.83 → 3.84 → 3.85
    """
    lines = ["## ETF 近期数据摘要", ""]

    for symbol, df in data.items():
        if df.empty:
            continue

        recent = df.tail(n_recent).copy()
        latest = recent.iloc[-1]

        ma20 = latest.get("ma20", np.nan)
        ma60 = latest.get("ma60", np.nan)
        ma5 = latest.get("ma5", np.nan)
        rsi = latest.get("rsi", np.nan)
        macd_hist = latest.get("macd_hist", np.nan)
        bb_pct = latest.get("bb_pct", np.nan)

        fund = fundamental.get(symbol, {})
        nav = fund.get("nav")
        premium = fund.get("premium")
        scale = fund.get("scale")

        close_series = recent["close"].tolist()
        price_str = " → ".join([f"{c:.2f}" for c in close_series])

        lines.append(f"### {symbol}")
        if fund.get("fund_name"):
            lines.append(f"  基金名称: {fund['fund_name']}")

        # 技术面
        tech_parts = []
        if not pd.isna(latest.get("close")):
            tech_parts.append(f"最新收盘: {latest['close']:.2f}")
        if not pd.isna(ma5):
            tech_parts.append(f"MA5: {ma5:.2f}")
        if not pd.isna(ma20):
            tech_parts.append(f"MA20: {ma20:.2f}")
        if not pd.isna(ma60):
            tech_parts.append(f"MA60: {ma60:.2f}")

        tech_line = " | ".join(tech_parts)
        if tech_line:
            lines.append(f"  {tech_line}")

        # 指标
        indicator_parts = []
        if not pd.isna(rsi):
            rsi_state = "超买" if rsi > 70 else ("超卖" if rsi < 30 else "中性")
            indicator_parts.append(f"RSI: {rsi:.1f}({rsi_state})")
        if not pd.isna(macd_hist):
            macd_state = "金叉" if macd_hist > 0 else "死叉"
            indicator_parts.append(f"MACD: {macd_hist:.3f}({macd_state})")
        if not pd.isna(bb_pct):
            bb_state = "上轨" if bb_pct > 0.8 else ("下轨" if bb_pct < 0.2 else "中轨")
            indicator_parts.append(f"布林带: {bb_pct:.1%}({bb_state})")

        ind_line = " | ".join(indicator_parts)
        if ind_line:
            lines.append(f"  {ind_line}")

        # 基本面
        fund_parts = []
        if nav is not None:
            fund_parts.append(f"净值: {nav:.2f}")
        if premium is not None:
            fund_parts.append(f"溢价率: {premium:.2%}")
        if scale is not None:
            fund_parts.append(f"规模: {scale:.1f}亿")
        if fund_parts:
            lines.append(f"  {' | '.join(fund_parts)}")

        # 走势
        lines.append(f"  近{n_recent}日: {price_str}")
        lines.append("")

    return "\n".join(lines)


def get_etf_summary_for_node(
    data: Dict[str, pd.DataFrame],
    node_type: str,
    n_recent: int = 5,
) -> str:
    """
    按节点类型生成精简的数据摘要。

    node_type: "tech" | "fundamental" | "sentiment"
    """
    if node_type == "tech":
        return _get_tech_summary(data, n_recent)
    elif node_type == "fundamental":
        return _get_fundamental_summary(data, n_recent)
    elif node_type == "sentiment":
        return _get_sentiment_summary(data, n_recent)
    return ""


def _get_tech_summary(data: Dict[str, pd.DataFrame], n: int) -> str:
    """技术面摘要"""
    lines = ["## 技术指标数据", ""]
    for symbol, df in data.items():
        if df.empty:
            continue
        recent = df.tail(n)
        latest = recent.iloc[-1]
        rsi = latest.get("rsi", np.nan)
        macd_hist = latest.get("macd_hist", np.nan)
        bb_pct = latest.get("bb_pct", np.nan)
        ma5 = latest.get("ma5", np.nan)
        ma20 = latest.get("ma20", np.nan)

        parts = [f"{symbol}:", f"  收盘价: {latest['close']:.2f}"]
        if not pd.isna(ma5) and not pd.isna(ma20):
            trend = "多头" if ma5 > ma20 else "空头"
            parts.append(f"  MA5({ma5:.2f}) vs MA20({ma20:.2f}): {trend}")
        if not pd.isna(rsi):
            parts.append(f"  RSI(14): {rsi:.1f}")
        if not pd.isna(macd_hist):
            parts.append(f"  MACD柱: {macd_hist:.3f}")
        if not pd.isna(bb_pct):
            parts.append(f"  布林带分位: {bb_pct:.1%}")
        parts.append("")
        lines.extend(parts)
    return "\n".join(lines)


def _get_fundamental_summary(data: Dict[str, pd.DataFrame], n: int) -> str:
    """基本面摘要（从数据中提取）"""
    lines = ["## 基本面数据", ""]
    for symbol, df in data.items():
        if df.empty:
            continue
        recent = df.tail(n)
        latest = recent.iloc[-1]
        close = latest.get("close", np.nan)
        vol = latest.get("volume", np.nan)
        vol_ma20 = recent["volume"].mean()
        vol_ratio = vol / vol_ma20 if vol_ma20 > 0 else np.nan

        parts = [f"{symbol}:", f"  最新价: {close:.2f}"]
        if not pd.isna(vol) and not pd.isna(vol_ratio):
            parts.append(f"  成交量: {vol:,.0f}（量比: {vol_ratio:.2f}）")
        parts.append("")
        lines.extend(parts)
    return "\n".join(lines)


def _get_sentiment_summary(data: Dict[str, pd.DataFrame], n: int) -> str:
    """情绪面摘要（从量价数据推断）"""
    lines = ["## 量价情绪数据", ""]
    for symbol, df in data.items():
        if df.empty:
            continue
        recent = df.tail(n).copy()
        recent["return"] = recent["close"].pct_change()
        latest = recent.iloc[-1]

        vol = latest.get("volume", np.nan)
        vol_avg = df.tail(20)["volume"].mean()
        vol_ratio = vol / vol_avg if vol_avg > 0 else np.nan
        n_return = (recent["close"].iloc[-1] / recent["close"].iloc[0]) - 1 if len(recent) > 1 else 0

        parts = [f"{symbol}:", f"  最新价: {latest['close']:.2f}"]
        if not pd.isna(vol_ratio):
            heat = "放量" if vol_ratio > 1.5 else ("缩量" if vol_ratio < 0.5 else "平量")
            parts.append(f"  成交量/20日均: {vol_ratio:.2f} ({heat})")
        parts.append(f"  近{n}日涨跌幅: {n_return:.2%}")

        recent_returns = recent["return"].dropna().tolist()
        if recent_returns:
            consecutive = 0
            direction = 1 if recent_returns[-1] > 0 else -1
            for r in reversed(recent_returns):
                if (r > 0) == (direction > 0):
                    consecutive += 1
                else:
                    break
            trend = "连涨" if direction > 0 else "连跌"
            parts.append(f"  近{n}日{trend}: {consecutive}天")
        parts.append("")
        lines.extend(parts)
    return "\n".join(lines)
